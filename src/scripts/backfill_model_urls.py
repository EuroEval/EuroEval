"""Backfill model_url field in existing processed results."""

from __future__ import annotations

import io
import json
import logging
import tarfile

from leaderboards.link_generation import generate_model_url
from leaderboards.paths import PROCESSED_RESULTS_DIR, RESULTS_PATH
from leaderboards.result_loading import load_raw_results

logger = logging.getLogger(__name__)


def get_model_name(record: dict) -> str:
    """Get model name from record, supporting both EEE and old formats.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        The model name.
    """
    if "model_info" in record:
        return record["model_info"]["name"]
    return record.get("model", "unknown")


def backfill_urls() -> None:
    """Load existing results, add model_url if missing, save back."""
    logger.info("Loading existing processed results...")
    results = load_raw_results()

    # Count how many records need URL backfilling
    records_needing_url = 0
    unique_models: set[str] = set()
    models_needing_url: set[str] = set()

    for record in results:
        model_name = get_model_name(record)
        unique_models.add(model_name)
        if "model_url" not in record or record.get("model_url") is None:
            records_needing_url += 1
            models_needing_url.add(model_name)

    logger.info(f"Found {len(results):,} total records")
    logger.info(f"Found {len(unique_models):,} unique models")
    logger.info(f"{records_needing_url:,} records need model_url backfilling")
    logger.info(f"{len(models_needing_url):,} models need URLs generated")

    if not models_needing_url:
        logger.info("All records already have model_url. Nothing to do.")
        return

    # Generate URLs for models that need them
    logger.info("Generating URLs for models...")
    model_url_cache: dict[str, str | None] = {}
    for model_name in models_needing_url:
        model_url_cache[model_name] = generate_model_url(model_id=model_name)

    # Add model_url to records
    updated_records = []
    for record in results:
        model_name = get_model_name(record)
        if "model_url" not in record or record.get("model_url") is None:
            record["model_url"] = model_url_cache.get(model_name)
        updated_records.append(record)

    # Group by model for storage
    records_by_model: dict[str, list[dict]] = {}
    for record in updated_records:
        model_name = get_model_name(record)
        # Strip anchor tags before creating filename
        model_name_clean = model_name.split("@")[0].split("#")[0]
        filename = model_name_clean.replace("/", "_").replace(".", "_") + ".jsonl"
        records_by_model.setdefault(filename, []).append(record)

    # Create backup before overwriting
    if RESULTS_PATH.exists():
        backup_path = RESULTS_PATH.with_suffix(".tar.gz.bak")
        logger.info(f"Creating backup at {backup_path}")
        backup_path.write_bytes(RESULTS_PATH.read_bytes())

    # Save to processed results directory
    logger.info("Saving updated records...")
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, records in records_by_model.items():
        file_path = PROCESSED_RESULTS_DIR / filename
        content = "\n".join(json.dumps(record) for record in records) + "\n"
        file_path.write_text(content, encoding="utf-8")

    # Rebuild results.tar.gz
    logger.info("Rebuilding results.tar.gz...")
    with tarfile.open(RESULTS_PATH, "w:gz") as tar:
        # Raw records
        raw_content_bytes = "\n".join(json.dumps(record) for record in results).encode(
            encoding="utf-8"
        )
        raw_tarinfo = tarfile.TarInfo(name="results/results.jsonl")
        raw_tarinfo.size = len(raw_content_bytes)
        raw_fileobj = io.BytesIO(raw_content_bytes)
        tar.addfile(tarinfo=raw_tarinfo, fileobj=raw_fileobj)

        # Processed records
        processed_content_bytes = "\n".join(
            json.dumps(record) for record in updated_records
        ).encode(encoding="utf-8")
        processed_tarinfo = tarfile.TarInfo(name="results/results.processed.jsonl")
        processed_tarinfo.size = len(processed_content_bytes)
        processed_fileobj = io.BytesIO(processed_content_bytes)
        tar.addfile(tarinfo=processed_tarinfo, fileobj=processed_fileobj)

    logger.info(f"Backfilled model_url for {len(models_needing_url):,} models")
    logger.info(f"Updated {records_needing_url:,} records")
    logger.info(f"Results saved to {RESULTS_PATH}")
    logger.info(f"Backup saved to {backup_path if RESULTS_PATH.exists() else 'N/A'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backfill_urls()
