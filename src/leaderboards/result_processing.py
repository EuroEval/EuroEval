"""Process EuroEval records into per-model result files.

This module is the orchestrator: it loads raw records, deduplicates them,
repairs and validates their metadata, and writes the processed results as one
JSON file per logical result, both locally in RESULTS_DIR and to the Hugging
Face results bucket.
"""

from __future__ import annotations

import json
import logging
import re
import typing as t
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi
from tqdm.auto import tqdm

from .cache import Cache
from .constants import HF_RESULTS_BUCKET, RESULTS_DIR
from .evaluation_common import resolve_hf_token
from .model_metadata import add_missing_entries, fix_metadata, record_is_valid
from .record_fields import deduplicate_records
from .records import get_model_name
from .result_identity import (
    ResultIdentity,
    dedup_newer_record,
    identity_from_eee_record,
    raise_on_collision,
    record_relative_path,
)
from .result_loading import load_raw_results

logger = logging.getLogger(__name__)


def process_results(
    min_version: str,
    min_number_of_model_records: int,
    banned_versions: list[str],
    banned_model_patterns: list[re.Pattern],
    api_model_patterns: list[re.Pattern],
    trained_from_scratch_patterns: list[re.Pattern],
) -> None:
    """Process EuroEval records from a JSONL file.

    Args:
        min_version:
            The minimum EuroEval version to include.
        min_number_of_model_records:
            The minimum number of records for a model to be included.
        banned_versions:
            A list of banned EuroEval versions to filter out.
        banned_model_patterns:
            A list of regex patterns to filter out models that should not be
            included.
        api_model_patterns:
            A list of regex patterns for API inference models.
        trained_from_scratch_patterns:
            A list of regex patterns for trained-from-scratch models.
    """
    # Load raw results first so per-model files in RESULTS_DIR are synced.
    records = load_raw_results()

    # Build the metadata cache from the synced per-model result files.
    cache = Cache.from_results_dir(RESULTS_DIR)
    num_raw_records = len(records)

    records = deduplicate_records(records=records)
    num_duplicates = num_raw_records - len(records)
    if num_duplicates:
        logger.info(f"Removed {num_duplicates:,} duplicates.")

    fixed_records = [
        fix_metadata(record=record)
        for record in tqdm(records, desc="Fixing metadata in records")
    ]

    processed_records = [
        record
        for record in fixed_records
        if record_is_valid(
            record=record,
            min_version=min_version,
            banned_versions=banned_versions,
            banned_model_patterns=banned_model_patterns,
            api_model_patterns=api_model_patterns,
        )
    ]

    # Remove records for models with few records
    counter = Counter([get_model_name(record) for record in processed_records])
    processed_records = [
        record
        for record in processed_records
        if counter[get_model_name(record)] >= min_number_of_model_records
    ]

    num_invalid_records = num_raw_records - num_duplicates - len(processed_records)
    if num_invalid_records > 0:
        logger.info(f"Removed {num_invalid_records:,} invalid records.")

    processed_records = [
        add_missing_entries(
            record=record,
            trained_from_scratch_patterns=trained_from_scratch_patterns,
            cache=cache,
        )
        for record in tqdm(processed_records, desc="Adding missing entries")
    ]

    _upload_per_model_files(processed_records=processed_records)


def _upload_per_model_files(processed_records: list[dict[str, t.Any]]) -> None:
    """Write one JSON file per logical result and sync to the HF bucket.

    Each record is written to
    ``results/<sanitise(model_id)>/<dataset>__<split>__<shot>.json``.
    If two records resolve to the same path, the newer one is kept (based on
    version and timestamp). Records with unresolvable model identities are dropped
    with a log line.

    Args:
        processed_records:
            The processed records to upload.

    Raises:
        RuntimeError:
            If HF_TOKEN is not set or bucket sync fails.
        ValueError:
            If distinct identities sanitise to the same relative path.
    """  # noqa: DOC502
    hf_token = resolve_hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Cannot upload results to Hugging Face bucket. "
            "Run 'hf auth login' or set the HF_TOKEN environment variable."
        )

    # Group records by path, handling deduplication and collision detection.
    # Store (record, identity) to detect when distinct identities map to same path.
    records_by_path: dict[Path, tuple[dict[str, t.Any], ResultIdentity]] = {}
    dropped_count = 0

    for record in processed_records:
        try:
            identity = identity_from_eee_record(record)
        except ValueError as e:
            logger.warning(f"Dropping record with unresolvable identity: {e}")
            dropped_count += 1
            continue

        relative_path = record_relative_path(
            model_id=identity[0],
            dataset=identity[1],
            validation_split=identity[2],
            few_shot=identity[3],
        )

        if relative_path in records_by_path:
            # Path collision - check if identities match
            existing_record, existing_identity = records_by_path[relative_path]
            # raise_on_collision will raise ValueError if identities differ
            raise_on_collision(existing_identity, identity)
            # Identities match - keep the newer record based on version/timestamp
            records_by_path[relative_path] = (
                dedup_newer_record(existing_record, record),
                identity,
            )
        else:
            records_by_path[relative_path] = (record, identity)

    hf_results_bucket = f"hf://buckets/{HF_RESULTS_BUCKET}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Track which files were written to upload only these files
    written_files: list[tuple[str | Path | bytes, str]] = []

    logger.info("Writing result files...")
    for relative_path, (record, _) in records_by_path.items():
        file_path = RESULTS_DIR / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Use canonical JSON serialization for consistent comparison
        canonical_content = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

        # Only write if file doesn't exist or content differs (semantic comparison)
        if file_path.exists():
            try:
                existing_content = file_path.read_text(encoding="utf-8").strip()
                # Parse and compare dicts for semantic equality
                existing_record = json.loads(existing_content)
                canonical_existing = json.dumps(
                    existing_record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                if canonical_existing == canonical_content:
                    continue  # Skip unchanged files
            except (json.JSONDecodeError, OSError):
                pass  # If we can't parse existing, overwrite it

        file_path.write_text(canonical_content + "\n", encoding="utf-8")
        # Skip empty files
        if file_path.stat().st_size > 0:
            written_files.append((file_path, str(relative_path)))

    api = HfApi()
    # Upload only the written files to the bucket
    api.batch_bucket_files(
        bucket_id=HF_RESULTS_BUCKET, add=written_files, token=hf_token
    )
    logger.info(f"Uploaded {len(written_files):,} result files to {hf_results_bucket}.")

    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count:,} records with unresolvable identities.")
