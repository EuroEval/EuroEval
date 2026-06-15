"""Backfill model_url field in existing processed results."""

from __future__ import annotations

import io
import json
import logging
import tarfile
from pathlib import Path

from huggingface_hub import HfApi
from leaderboards.link_generation import generate_model_url
from leaderboards.paths import PROCESSED_RESULTS_DIR, RESULTS_PATH

logger = logging.getLogger(__name__)


def _model_id_to_filename(model_id: str) -> str:
    """Convert a model ID to a safe filename.

    Matches the convention in collect_evaluation_results.py.

    Args:
        model_id:
            The model identifier (e.g., "meta-llama/Llama-3-8B").

    Returns:
        A safe filename with slashes and dots replaced by underscores.
    """
    return model_id.replace("/", "_").replace(".", "_") + ".jsonl"


def backfill_urls() -> None:
    """Load existing processed results, add model_url if missing, save back."""
    logger.info("Loading existing processed results from per-model files...")

    # Load all per-model processed result files
    model_files = sorted(PROCESSED_RESULTS_DIR.glob("*.jsonl"))
    if not model_files:
        logger.error(f"No processed result files found in {PROCESSED_RESULTS_DIR}")
        return

    logger.info(f"Found {len(model_files):,} model files")

    # Process each model file
    files_updated = 0
    models_needing_url = 0
    total_records = 0

    for model_file in model_files:
        records = []
        file_had_missing_urls = False

        with open(model_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                total_records += 1

                # Add model_url if missing
                if "model_url" not in record or record.get("model_url") is None:
                    file_had_missing_urls = True
                    model_id = record.get("model", "unknown")
                    try:
                        record["model_url"] = generate_model_url(model_id=model_id)
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate URL for {model_id}: {e}. "
                            "Setting model_url to None."
                        )
                        record["model_url"] = None

                records.append(record)

        if file_had_missing_urls:
            models_needing_url += 1
            # Write updated records back
            content = "\n".join(json.dumps(record) for record in records) + "\n"
            model_file.write_text(content, encoding="utf-8")
            files_updated += 1

    logger.info(f"Processed {total_records:,} records from {len(model_files):,} files")
    logger.info(f"{models_needing_url:,} models needed URL backfilling")
    logger.info(f"{files_updated:,} files updated")

    if not files_updated:
        logger.info("All records already have model_url. Nothing to do.")
        return

    # Create backup before overwriting tar.gz
    if RESULTS_PATH.exists():
        backup_path = RESULTS_PATH.with_name(RESULTS_PATH.name + ".bak")
        logger.info(f"Creating backup at {backup_path}")
        backup_path.write_bytes(RESULTS_PATH.read_bytes())

    # Rebuild results.tar.gz from updated per-model files
    logger.info("Rebuilding results.tar.gz from per-model files...")

    # Collect all records
    all_records = []
    for model_file in sorted(PROCESSED_RESULTS_DIR.glob("*.jsonl")):
        with open(model_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    logger.info(f"Collected {len(all_records):,} total records")

    with tarfile.open(RESULTS_PATH, "w:gz") as tar:
        # Raw records - need to load from raw-results bucket or skip
        # For now, we'll just include processed records
        # Raw records can be synced separately via hf sync

        # Processed records
        processed_content_bytes = "\n".join(
            json.dumps(record) for record in all_records
        ).encode(encoding="utf-8")
        processed_tarinfo = tarfile.TarInfo(name="results/results.processed.jsonl")
        processed_tarinfo.size = len(processed_content_bytes)
        processed_fileobj = io.BytesIO(processed_content_bytes)
        tar.addfile(tarinfo=processed_tarinfo, fileobj=processed_fileobj)

    logger.info(f"Backfilled model_url for {models_needing_url:,} models")
    logger.info(f"Updated {files_updated:,} files")
    logger.info(f"Results saved to {RESULTS_PATH}")
    logger.info(f"Backup saved to {backup_path if RESULTS_PATH.exists() else 'N/A'}")

    # Sync back to HF bucket
    try:
        api = HfApi()
        hf_processed_bucket = "hf://buckets/EuroEval/processed-results"
        logger.info(f"Syncing updated results to {hf_processed_bucket}...")
        api.sync_bucket(source=str(PROCESSED_RESULTS_DIR), dest=hf_processed_bucket)
        logger.info("Successfully synced to HF bucket.")
    except Exception as e:
        logger.warning(f"Failed to sync to HF bucket: {e}")
        logger.warning("Run 'hf sync' manually to upload to HF bucket.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backfill_urls()
