"""Restore valuable metadata from backup without losing model_url."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import HfApi

from leaderboards.paths import PROCESSED_RESULTS_DIR

logger = logging.getLogger(__name__)

# Backup location
BACKUP_PROCESSED_DIR = Path("/tmp/processed_backup/EuroEval/results/processed")

# Fields to restore from backup
METADATA_FIELDS = ["commercially_licensed", "open", "trained_from_scratch"]


def restore_metadata() -> None:
    """Restore valuable metadata fields from backup while keeping model_url."""
    logger.info(f"Loading backup metadata from {BACKUP_PROCESSED_DIR}...")

    if not BACKUP_PROCESSED_DIR.exists():
        logger.error(f"Backup directory not found: {BACKUP_PROCESSED_DIR}")
        return

    # Load metadata from backup files
    backup_metadata: dict[str, dict] = {}
    backup_files = list(BACKUP_PROCESSED_DIR.glob("*.jsonl"))
    logger.info(f"Found {len(backup_files):,} files in backup")

    for model_file in backup_files:
        try:
            with open(model_file, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if not first_line.strip():
                    continue
                record = json.loads(first_line)
                model_id = record.get("model", "unknown")
                # Extract only the metadata fields we need
                metadata = {field: record.get(field) for field in METADATA_FIELDS}
                backup_metadata[model_id] = metadata
        except Exception as e:
            logger.warning(f"Failed to read {model_file.name}: {e}")

    logger.info(f"Loaded metadata for {len(backup_metadata):,} models from backup")

    # Update current processed results
    logger.info("Updating current processed results with backup metadata...")
    current_files = list(PROCESSED_RESULTS_DIR.glob("*.jsonl"))
    logger.info(f"Found {len(current_files):,} current model files")

    files_updated = 0
    records_updated = 0

    for model_file in current_files:
        updated_records = []
        file_was_updated = False

        with open(model_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                model_id = record.get("model", "unknown")

                # Check if we have backup metadata for this model
                if model_id in backup_metadata:
                    # Restore metadata fields, but keep existing model_url
                    for field in METADATA_FIELDS:
                        if field in backup_metadata[model_id]:
                            old_value = record.get(field)
                            new_value = backup_metadata[model_id][field]
                            if old_value != new_value:
                                record[field] = new_value
                                file_was_updated = True
                                records_updated += 1

                updated_records.append(record)

        if file_was_updated:
            # Write updated records back
            content = "\n".join(json.dumps(record) for record in updated_records) + "\n"
            model_file.write_text(content, encoding="utf-8")
            files_updated += 1

    logger.info(f"Updated {records_updated:,} records in {files_updated:,} files")
    logger.info("Preserved model_url and other existing fields")

    # Sync back to HF bucket
    try:
        api = HfApi()
        hf_processed_bucket = "hf://buckets/EuroEval/processed-results"
        logger.info(f"Syncing restored results to {hf_processed_bucket}...")
        api.sync_bucket(source=str(PROCESSED_RESULTS_DIR), dest=hf_processed_bucket)
        logger.info("Successfully synced to HF bucket.")
    except Exception as e:
        logger.warning(f"Failed to sync to HF bucket: {e}")
        logger.warning("Run 'hf sync' manually to upload to HF bucket.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    restore_metadata()
