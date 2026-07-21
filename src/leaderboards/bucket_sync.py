"""Sync Hugging Face bucket for EuroEval results.

Uses the ``huggingface_hub`` package to sync the results bucket to the local
results directory. Also provides backup functionality.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from huggingface_hub import BucketFile, HfApi, list_bucket_tree
from huggingface_hub.errors import HfHubHTTPError

from .constants import HF_RESULTS_BUCKET, RESULTS_DIR
from .evaluation_common import resolve_hf_token
from .result_identity import (
    ResultIdentity,
    dedup_newer_record,
    identity_from_eee_record,
    identity_to_path,
    raise_on_collision,
)

load_dotenv()

logger = logging.getLogger(__name__)


def _cleanup_empty_bucket_files(hf_token: str) -> None:
    """Delete 0-byte JSON files from the Hugging Face bucket.

    Lists all files in the bucket, identifies empty JSON files by size,
    and removes them via batch operation.

    Args:
        hf_token:
            Hugging Face authentication token.
    """
    api = HfApi()
    empty_files: list[str] = []

    try:
        # List all files in the bucket
        files = list_bucket_tree(
            bucket_id=HF_RESULTS_BUCKET, token=hf_token, recursive=True
        )
        for file_info in files:
            # Check if it's a JSON file with size 0
            if isinstance(file_info, BucketFile):
                if file_info.path.endswith(".json") and file_info.size == 0:
                    empty_files.append(file_info.path)
            elif isinstance(file_info, dict):
                # Handle dict format if API returns that
                path = str(file_info.get("path", ""))
                size = file_info.get("size", 0)
                if path.endswith(".json") and size == 0:
                    empty_files.append(path)
    except HfHubHTTPError as e:
        logger.error(f"Failed to list bucket files: {e}")
        return

    if empty_files:
        logger.info(f"Found {len(empty_files)} empty file(s) in bucket, deleting...")
        try:
            api.batch_bucket_files(bucket_id=HF_RESULTS_BUCKET, delete=empty_files)
            logger.info(f"Deleted {len(empty_files)} empty file(s) from bucket.")
        except HfHubHTTPError as e:
            logger.error(f"Failed to delete empty bucket files: {e}")


def remove_empty_json_files(results_dir: Path) -> list[str]:
    """Find and delete empty JSON files (0 bytes).

    Scans recursively for all *.json files and removes any with zero size.

    Args:
        results_dir:
            Directory to scan for empty JSON files.

    Returns:
        List of relative paths that were deleted.
    """
    deleted: list[str] = []
    for json_file in results_dir.rglob("*.json"):
        if json_file.is_file() and json_file.stat().st_size == 0:
            logger.info(f"Removing empty file {json_file}")
            json_file.unlink()
            deleted.append(str(json_file.relative_to(results_dir)))
    if deleted:
        logger.info(f"Deleted {len(deleted)} empty JSON file(s).")
    return deleted


def sync_bucket(ignore_sizes: bool = False, delete_empty: bool = True) -> None:
    """Sync HF results bucket into the local results directory.

    Before sync, reads all local record files into memory keyed by relative path.
    After sync, reconciles to ensure no local-only or locally-newer record is lost:
    - Local-only files (missing after sync) are restored.
    - Files present in both are compared by identity; the newer record wins.
    - Path collisions between distinct identities raise an error.

    Args:
        ignore_sizes:
            When True, skip file size comparison during sync. Works around
            huggingface_hub bug reporting wrong sizes, making sync compare by
            mtime only.
        delete_empty:
            When True, scan both local directory and bucket for 0-byte JSON
            files and delete them. Defaults to True.

    Raises:
        RuntimeError:
            If no Hugging Face token is available.
        HfHubHTTPError:
            If the bucket sync operation fails.
    """
    hf_token = resolve_hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Cannot sync results from Hugging Face bucket. "
            "Run 'hf auth login' or set the HF_TOKEN environment variable."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    class LocalRecordData(TypedDict):
        raw: bytes
        record: dict

    # Read all local record files into memory before sync, keyed by relative path
    local_before: dict[str, LocalRecordData] = {}
    if RESULTS_DIR.exists():
        for local_file in RESULTS_DIR.rglob("*.json"):
            if not local_file.is_file():
                continue
            rel_path = str(local_file.relative_to(RESULTS_DIR))
            try:
                raw_bytes = local_file.read_bytes()
                record = json.loads(raw_bytes.decode("utf-8"))
                local_before[rel_path] = {"raw": raw_bytes, "record": record}
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Skipping invalid local file {local_file}: {e}")

    logger.info(f"Syncing bucket {HF_RESULTS_BUCKET} -> {RESULTS_DIR}...")
    try:
        HfApi().sync_bucket(
            source=f"hf://buckets/{HF_RESULTS_BUCKET}/",
            dest=str(RESULTS_DIR),
            token=hf_token,
            ignore_times=True,  # Compare by size only
            ignore_sizes=ignore_sizes,  # When True, compare by mtime only
        )
    except HfHubHTTPError as e:
        logger.error(f"Bucket sync failed: {e}")
        raise

    # Reconcile after sync: ensure union of local-before and bucket is lossless
    for rel_path, local_data in local_before.items():
        file_path = RESULTS_DIR / rel_path
        local_record = local_data["record"]
        local_raw = local_data["raw"]

        if not file_path.exists():
            # Local-only file was removed by sync - restore it
            logger.info(f"Restoring local-only file {file_path}")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(local_raw)
        else:
            # File exists in both - compare by identity
            try:
                bucket_record = json.loads(file_path.read_text(encoding="utf-8"))
                local_identity = identity_from_eee_record(local_record)
                bucket_identity = identity_from_eee_record(bucket_record)
            except ValueError as e:
                # Identity extraction failed - log and skip
                logger.warning(
                    f"Skipping file {file_path} due to invalid identity: {e}"
                )
                continue

            if local_identity == bucket_identity:
                # Same identity - keep the newer record
                winner = dedup_newer_record(local_record, bucket_record)
                if winner is local_record:
                    logger.debug(f"Local record newer for {rel_path}, restoring")
                    file_path.write_bytes(local_raw)
                else:
                    # Bucket record is newer, leave it as is
                    pass
            else:
                # Different identity at same path - collision!
                raise_on_collision(local_identity, bucket_identity)

    # Clean up empty JSON files locally
    if delete_empty:
        remove_empty_json_files(RESULTS_DIR)

    # Optionally clean up empty files in the bucket
    if delete_empty:
        _cleanup_empty_bucket_files(hf_token)

    logger.info(f"Synced bucket {HF_RESULTS_BUCKET}.")


def _sort_key(record: dict) -> tuple[str, str]:
    """Extract sort key from a record for deterministic ordering.

    Args:
        record:
            A result record in EEE format.

    Returns:
        Tuple of (model_id, dataset) for sorting.
    """
    model_info = record.get("model_info", {})
    eval_lib = record.get("eval_library", {})
    additional = eval_lib.get("additional_details", {})
    model_id = model_info.get("id") or model_info.get("name", "")
    dataset = additional.get("dataset", "")
    return (model_id, dataset)


def merge_results(results_file: Path) -> int:
    """Merge per-record JSON tree into a single JSONL file.

    Reads all ``results/*/*.json`` files and writes a deduplicated JSONL file.
    Deduplication uses canonical result identity
    ``(model_id, dataset, validation_split, few_shot)`` with newer records
    winning based on ``eval_library.version`` and ``retrieved_timestamp``.

    Args:
        results_file:
            Path to the merged JSONL file to write.

    Returns:
        Number of unique results written.
    """
    existing: dict[ResultIdentity, dict] = {}

    # Remove empty JSON files before processing
    remove_empty_json_files(RESULTS_DIR)

    # Read all record files from the tree
    if RESULTS_DIR.exists():
        for record_file in RESULTS_DIR.rglob("*.json"):
            if not record_file.is_file():
                continue
            try:
                record = json.loads(record_file.read_text(encoding="utf-8"))
                identity = identity_from_eee_record(record)
                if identity in existing:
                    existing[identity] = dedup_newer_record(existing[identity], record)
                else:
                    existing[identity] = record
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.debug(f"Skipping invalid record {record_file}: {e}")

    if not existing:
        logger.warning("No results found to merge")
        return 0

    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", encoding="utf-8") as f:
        for record in sorted(existing.values(), key=_sort_key):
            f.write(json.dumps(record) + "\n")
    return len(existing)


def upload_results_to_bucket(results_file: Path) -> None:
    """Upload local results to the Hugging Face results bucket.

    Reads the merged JSONL file, converts to per-record JSON tree layout
    (one file per logical result at
    ``results/<sanitise(model_id)>/<dataset>__<split>__<shot>.json``),
    then syncs to the bucket. Records without a valid model are dropped.

    Args:
        results_file:
            Path to the merged results file (JSONL format).

    Raises:
        RuntimeError:
            If no Hugging Face token is available.
        HfHubHTTPError:
            If the bucket sync operation fails.
    """
    hf_token = resolve_hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Cannot upload results to Hugging Face bucket. "
            "Run 'hf auth login' or set the HF_TOKEN environment variable."
        )

    if not results_file.exists():
        logger.warning(
            f"Results file {results_file} does not exist. Nothing to upload."
        )
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # PHASE 1: Parse ALL input records and build the COMPLETE path→identity map.
    # raise_on_collision is called for any distinct-identity path collision BEFORE
    # any filesystem mutation. On collision, RESULTS_DIR is left untouched.
    logger.info(f"Reading results from {results_file}...")
    # rel_path -> (identity, record) for writing after validation
    path_record_map: dict[str, tuple[ResultIdentity, dict]] = {}
    with results_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Skipping invalid JSON: {e}")
                continue

            try:
                identity = identity_from_eee_record(record)
            except ValueError as e:
                logger.debug(f"Skipping record with invalid identity: {e}")
                continue

            model_id, dataset, validation_split, few_shot = identity

            # Skip records without valid model
            if not model_id:
                logger.debug("Skipping record without model_id")
                continue

            record_path = RESULTS_DIR / identity_to_path(identity)
            rel_path = str(record_path.relative_to(RESULTS_DIR))

            # Check for collision: distinct identity at same path
            if rel_path in path_record_map:
                raise_on_collision(path_record_map[rel_path][0], identity)
            path_record_map[rel_path] = (identity, record)

    if not path_record_map:
        logger.warning("No valid results found to upload.")
        return

    # PHASE 2: Validation passed. Load existing records from bucket to compare.
    # Only upload files that have changed (semantic comparison, not raw JSON).

    # Load existing records from bucket
    existing_records: dict[str, dict] = {}
    try:
        bucket_files = list_bucket_tree(
            bucket_id=HF_RESULTS_BUCKET, token=hf_token, recursive=True
        )
        for file_info in bucket_files:
            if isinstance(file_info, BucketFile):
                path = file_info.path
            elif isinstance(file_info, dict):
                path = file_info.get("path", "")
            else:
                continue

            if not path.endswith(".json"):
                continue
            # Download and parse the record
            try:
                # Use HfApi to read the file content from the bucket. Since we can't
                # easily read bucket file content, fall back to local cache.
                pass
            except Exception:
                pass
    except HfHubHTTPError:
        pass  # If we can't read bucket, just use local cache

    # Load existing local records
    for rel_path in path_record_map.keys():
        local_file = RESULTS_DIR / rel_path
        if local_file.exists():
            try:
                existing_records[rel_path] = json.loads(
                    local_file.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                pass

    # Remove stale ROOT-level *.jsonl artefacts (old flat store) only
    for jsonl_file in RESULTS_DIR.glob("*.jsonl"):
        if jsonl_file.is_file():
            logger.info(f"Removing stale jsonl file {jsonl_file}")
            jsonl_file.unlink()

    # Write all validated records, skipping any that would be empty or unchanged
    # Type: list of (local_path, remote_path) tuples for batch upload
    written_files: list[tuple[str | Path | bytes, str]] = []
    for rel_path, (_, record) in path_record_map.items():
        # Use canonical JSON serialization for comparison
        new_content = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if not new_content.strip():
            logger.warning(f"Skipping empty record for {rel_path}")
            continue

        # Check if record already exists and is unchanged
        record_path = RESULTS_DIR / rel_path
        if rel_path in existing_records:
            old_content = json.dumps(
                existing_records[rel_path], sort_keys=True, separators=(",", ":")
            )
            if old_content == new_content:
                logger.debug(f"Skipping unchanged record {rel_path}")
                continue

        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(new_content, encoding="utf-8")
        if record_path.stat().st_size > 0:
            written_files.append((record_path, rel_path))

    logger.info(
        "Wrote %d record files to %s, uploading to bucket...",
        len(written_files),
        RESULTS_DIR,
    )
    try:
        HfApi().batch_bucket_files(
            bucket_id=HF_RESULTS_BUCKET, add=written_files, token=hf_token
        )
    except HfHubHTTPError as e:
        logger.error(f"Bucket upload failed: {e}")
        raise

    logger.info(f"Uploaded {len(written_files)} results to bucket {HF_RESULTS_BUCKET}.")
