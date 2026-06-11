"""Loading of results, to be converted into leaderboards."""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
import tarfile
import typing as t
from functools import cache
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from .backup import restore_from_backup_if_missing
from .hf_mount import MOUNT_POINT, hf_mount_context, is_hf_mount_available
from .paths import NEW_RESULTS_PATH, RAW_RESULTS_DIR, RESULTS_PATH

logger = logging.getLogger(__name__)

HF_RAW_BUCKET = "hf://buckets/EuroEval/raw-results"


def _sync_results_from_bucket() -> None:
    """Sync results from HF bucket and rebuild results.tar.gz.

    Uses hf-mount if available (live mount, no sync needed), otherwise
    falls back to huggingface_hub bucket sync.
    """
    # Prefer hf-mount if available
    if is_hf_mount_available():
        _sync_via_hf_mount()
    else:
        logger.info("hf-mount not available, using huggingface_hub sync.")
        _sync_via_huggingface_hub()


def _sync_via_hf_mount() -> None:
    """Sync via hf-mount daemon or fall back to existing cache.

    Tries hf-mount first. If the mount has fewer files than the
    local cache, the cache is authoritative and is used instead.

    Raises:
        FileNotFoundError:
            If neither mount nor cache exist.
    """
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Count files in existing cache (authoritative)
    cache_file_count = len(list(RAW_RESULTS_DIR.glob("*.jsonl")))

    # Try hf-mount first
    mount_point = MOUNT_POINT
    if mount_point.exists():
        file_count = 0
        for jsonl_file in mount_point.glob("*.jsonl"):
            dest = RAW_RESULTS_DIR / jsonl_file.name
            shutil.copy2(jsonl_file, dest)
            file_count += 1

        # Use mount if it has at least as many files as the cache
        if file_count >= cache_file_count:
            logger.info(f"Copied {file_count:,} model files from hf-mount daemon.")
            return
        elif cache_file_count > 0:
            logger.info(
                f"hf-mount has {file_count:,} files vs {cache_file_count:,} in cache. "
                "Using cache."
            )
        else:
            logger.info(f"hf-mount has {file_count:,} files (no cache to compare).")

    # Fall back to existing cache
    if cache_file_count > 0:
        logger.info(
            f"Using existing cache with {cache_file_count:,} model files "
            f"(from {RAW_RESULTS_DIR})."
        )
        return

    raise FileNotFoundError(
        "No results available. Start hf-mount or populate results/raw/"
    )


def _sync_via_huggingface_hub() -> None:
    """Fallback: sync via huggingface_hub (older, slower)."""
    # First try to restore from local backup if archive is missing
    if not RESULTS_PATH.exists():
        if restore_from_backup_if_missing():
            logger.info("Restored results.tar.gz from backup.")
        else:
            logger.warning("No backup found. Will rely on HF bucket.")

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results from archive first (if it exists)
    archive_lines: set[str] = set()
    if RESULTS_PATH.exists():
        try:
            with tarfile.open(RESULTS_PATH, "r:gz") as tar:
                results_file = tar.extractfile(member="results/results.jsonl")
                if results_file is not None:
                    archive_lines = {
                        line
                        for line in results_file.read()
                        .decode(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                    }
            logger.info(f"Loaded {len(archive_lines):,} existing results from archive.")
        except Exception as e:
            logger.warning(f"Could not load archive: {e}. Starting fresh.")

    # Sync from bucket
    try:
        logger.info(f"Syncing results from {HF_RAW_BUCKET}...")
        HfApi().sync_bucket(source=HF_RAW_BUCKET + "/", dest=str(RAW_RESULTS_DIR))
        logger.info("Downloaded results from bucket.")
    except HfHubHTTPError as e:
        logger.warning(f"Could not sync from bucket: {e}. Using existing archive only.")
        return

    # Load all per-model files from bucket cache
    bucket_lines: set[str] = set()
    for model_file in RAW_RESULTS_DIR.glob("*.jsonl"):
        lines = {
            line
            for line in model_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        bucket_lines.update(lines)

    n_files = len(list(RAW_RESULTS_DIR.glob("*.jsonl")))
    logger.info(
        f"Loaded {len(bucket_lines):,} results from {n_files:,} model files in bucket."
    )

    # Detect bucket staleness (bucket has fewer results than archive)
    if archive_lines and len(bucket_lines) < len(archive_lines):
        stale_count = len(archive_lines) - len(bucket_lines)
        logger.warning(
            f"Bucket appears stale: {stale_count:,} results in archive not in bucket. "
            f"Archive: {len(archive_lines):,}, Bucket: {len(bucket_lines):,}. "
            "Next successful upload_to_hf() will sync the bucket."
        )

    # Warn if bucket seems suspiciously small (no archive to compare against)
    if not archive_lines and len(bucket_lines) < 1000:
        logger.error(
            "WARNING: Only %d results in bucket, but no local archive "
            "to compare against. Expected 77,000+. If this is a fresh "
            "machine, verify the bucket is complete by checking a backup "
            "or another machine.",
            len(bucket_lines),
        )

    # Merge: archive + bucket, deduplicated by full line (which includes record hash)
    # Archive is preferred source of truth if they diverge
    all_lines = archive_lines | bucket_lines
    logger.info(
        "Merged %d unique results (archive: %d, bucket: %d).",
        len(all_lines),
        len(archive_lines),
        len(bucket_lines),
    )

    # Rebuild results.tar.gz with merged results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(RESULTS_PATH, "w:gz") as tar:
        # Raw records
        raw_content_bytes = "\n".join(sorted(all_lines)).encode(encoding="utf-8")
        raw_tarinfo = tarfile.TarInfo(name="results/results.jsonl")
        raw_tarinfo.size = len(raw_content_bytes)
        raw_fileobj = io.BytesIO(raw_content_bytes)
        tar.addfile(tarinfo=raw_tarinfo, fileobj=raw_fileobj)

        # Processed records (empty initially, will be filled by process_results)
        processed_content_bytes = b""
        processed_tarinfo = tarfile.TarInfo(name="results/results.processed.jsonl")
        processed_tarinfo.size = len(processed_content_bytes)
        processed_fileobj = io.BytesIO(processed_content_bytes)
        tar.addfile(tarinfo=processed_tarinfo, fileobj=processed_fileobj)

    logger.info(f"Rebuilt {RESULTS_PATH} with {len(all_lines):,} merged results.")


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
