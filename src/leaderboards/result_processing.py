"""Process EuroEval records from a JSONL file into per-model archives.

This module is the orchestrator: it loads raw records, deduplicates them,
repairs and validates their metadata, and writes the processed results both
to the Hugging Face results bucket (one file per model) and to the local
``results.tar.gz`` archive.
"""

from __future__ import annotations

import io
import logging
import re
import tarfile
import typing as t
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi
from tqdm.auto import tqdm

from .cache import Cache
from .constants import HF_RESULTS_BUCKET, RESULTS_DIR, RESULTS_PATH
from .eee_validation import dump_jsonl_records, validate_eee_records
from .model_metadata import add_missing_entries, fix_metadata, record_is_valid
from .records import get_model_name, get_record_hash, plain_model_id
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
    results_path = RESULTS_PATH

    # Build the cache from the results directory if available,
    # otherwise fall back to the compressed results file
    cache = Cache.from_processed_records(
        compressed_results_path=results_path, results_dir=RESULTS_DIR
    )

    records = load_raw_results()
    num_raw_records = len(records)

    records = _deduplicate_records(records=records)
    num_duplicates = num_raw_records - len(records)
    if num_duplicates:
        logger.info(f"Removed {num_duplicates:,} duplicates.")

    # Add missing metadata to records. If the metadata cannot be fixed, the
    # record is replaced with None, which is dropped below.
    fixed_records: list[dict[str, t.Any] | None] = [
        fix_metadata(record=record, cache=cache)
        for record in tqdm(records, desc="Fixing metadata in records")
    ]

    processed_records = [
        record
        for record in fixed_records
        if record is not None
        and record_is_valid(
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

    validate_eee_records(records=processed_records, context="processed results")

    _upload_per_model_files(processed_records=processed_records)
    _write_results_archive(
        processed_records=processed_records, results_path=results_path
    )


def _deduplicate_records(records: list[dict[str, t.Any]]) -> list[dict[str, t.Any]]:
    """Deduplicate records by hash, keeping the newest EuroEval version per hash.

    Args:
        records:
            The raw records.

    Returns:
        The deduplicated records.
    """
    all_hash_values = [get_record_hash(record=dct) for dct in records]
    unique_hash_values = sorted(set(all_hash_values))
    new_records = []
    for unique_hash_value in tqdm(unique_hash_values, desc="Processing records"):
        matches = [
            record
            for record, hash_value in zip(records, all_hash_values)
            if hash_value == unique_hash_value
        ]
        versions = [
            list(
                map(
                    int,
                    re.sub(
                        pattern=r"\.dev[0-9]+",
                        repl="",
                        string=match.get(
                            "euroeval_version", match.get("scandeval_version")
                        )
                        or "0.0.0",
                    ).split("."),
                )
            )
            for match in matches
        ]
        newest_version = max(versions)
        matches_with_newest_version = [
            match
            for match, version in zip(matches, versions)
            if version == newest_version
        ]
        new_records.append(matches_with_newest_version[-1])
    return new_records


def _upload_per_model_files(processed_records: list[dict[str, t.Any]]) -> None:
    """Group records into one per-model file and sync them to the HF bucket.

    Files are named by the model id with slashes replaced by underscores
    (dots preserved).

    Args:
        processed_records:
            The processed records to upload.
    """
    validate_eee_records(records=processed_records, context="per-model upload")

    results_by_model: dict[str, list[dict]] = {}
    for record in processed_records:
        model_id = record.get("model_info", {}).get("name") or record.get(
            "model", "unknown"
        )
        model_id_str = plain_model_id(model_id)
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


def _write_results_archive(
    processed_records: list[dict[str, t.Any]], results_path: Path
) -> None:
    """Write all processed records to the results.tar.gz archive.

    The archive stores all processed records with metadata as a single JSONL
    file.

    Args:
        processed_records:
            The processed records to archive.
        results_path:
            The path of the tar.gz archive to write.
    """
    content = dump_jsonl_records(records=processed_records)
    with tarfile.open(results_path, "w:gz") as tar:
        all_content_bytes = content.encode(encoding="utf-8")
        tarinfo = tarfile.TarInfo(name="results/results.jsonl")
        tarinfo.size = len(all_content_bytes)
        fileobj = io.BytesIO(all_content_bytes)
        tar.addfile(tarinfo=tarinfo, fileobj=fileobj)
