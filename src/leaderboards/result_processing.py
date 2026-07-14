"""Process EuroEval records into per-model result files.

This module is the orchestrator: it loads raw records, deduplicates them,
repairs and validates their metadata, and writes the processed results as one
JSONL file per model, both locally in RESULTS_DIR and to the Hugging Face
results bucket.
"""

from __future__ import annotations

import logging
import re
import typing as t
from collections import Counter

from huggingface_hub import HfApi
from tqdm.auto import tqdm

from .cache import Cache
from .constants import HF_RESULTS_BUCKET, RESULTS_DIR
from .eee_validation import dump_jsonl_records
from .model_metadata import add_missing_entries, fix_metadata, record_is_valid
from .record_fields import deduplicate_records
from .records import get_model_name, plain_model_id
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
    """Group records into one per-model file and sync them to the HF bucket.

    Files are named by the model id with slashes replaced by underscores
    (dots preserved).

    Args:
        processed_records:
            The processed records to upload.
    """
    results_by_model: dict[str, list[dict]] = {}
    for record in processed_records:
        model_id_str = plain_model_id(get_model_name(record))
        filename = model_id_str.replace("/", "_") + ".jsonl"
        results_by_model.setdefault(filename, []).append(record)

    hf_results_bucket = f"hf://buckets/{HF_RESULTS_BUCKET}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Uploading results to HF bucket...")
    for filename, model_records in results_by_model.items():
        file_path = RESULTS_DIR / filename
        content = dump_jsonl_records(records=model_records)
        file_path.write_text(content, encoding="utf-8")

    try:
        api = HfApi()
        api.sync_bucket(source=str(RESULTS_DIR), dest=hf_results_bucket)
        logger.info(
            f"Uploaded {len(results_by_model):,} model files to {hf_results_bucket}."
        )
    except Exception as e:
        logger.warning(f"Failed to sync results: {e}")
