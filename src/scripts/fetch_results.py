"""Fetch and merge benchmark results from the Hugging Face bucket.

Syncs the raw-results bucket and merges all results into
euroeval_benchmark_results.jsonl, preserving existing local results.
"""

import json
import logging
from pathlib import Path

from euroeval.leaderboards.bucket_sync import sync_bucket
from euroeval.leaderboards.paths import RAW_RESULTS_DIR

logger = logging.getLogger(__name__)

RESULTS_FILE = Path("euroeval_benchmark_results.jsonl")


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    sync_bucket()

    n_results = merge_results()
    if n_results > 0:
        logger.info("Merged %s unique results into %s", f"{n_results:,}", RESULTS_FILE)


def merge_results() -> int:
    """Merge all results into euroeval_benchmark_results.jsonl.

    Deduplicates by (model_id, dataset, validation_split, few_shot) key,
    which uniquely identifies an evaluation configuration.

    Local results are preserved; bucket results fill in gaps.

    Reuses similar logic to _rebuild_results_tar_gz(), but outputs to
    euroeval_benchmark_results.jsonl instead of results.tar.gz.

    Returns:
        Number of unique results written.
    """
    existing: dict[tuple, str] = {}

    # Load existing local results
    if RESULTS_FILE.exists():
        logger.info("Loading existing results from %s...", RESULTS_FILE)
        with RESULTS_FILE.open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = _extract_dedup_key(rec)
                    if key:
                        existing[key] = line.strip()
        logger.info("Found %s existing results", f"{len(existing):,}")

    # Load results from raw bucket
    if RAW_RESULTS_DIR.exists():
        logger.info("Loading results from %s...", RAW_RESULTS_DIR)
        bucket_count = 0
        for jsonl_file in sorted(RAW_RESULTS_DIR.glob("*.jsonl")):
            with jsonl_file.open() as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        key = _extract_dedup_key(rec)
                        if key:
                            existing[key] = line.strip()
                            bucket_count += 1
        logger.info("Loaded %s results from bucket", f"{bucket_count:,}")

    if not existing:
        logger.warning("No results found to merge")
        return 0

    # Write merged results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w") as f:
        for line in sorted(existing.values()):
            f.write(line + "\n")

    return len(existing)


def _extract_dedup_key(rec: dict) -> tuple | None:
    """Extract deduplication key from an evaluation result record.

    Key consists of:
    - model_id: from model_info.id
    - dataset: from eval_library.additional_details.dataset
    - validation_split: from eval_library.additional_details.validation_split
    - few_shot: from eval_library.additional_details.few_shot

    Args:
        rec: Evaluation result record as dictionary.

    Returns:
        Tuple key for deduplication, or None if required fields are missing.
    """
    try:
        model_id = rec.get("model_info", {}).get("id")
        eval_details = rec.get("eval_library", {}).get("additional_details", {})
        dataset = eval_details.get("dataset")
        validation_split = eval_details.get("validation_split")
        few_shot = eval_details.get("few_shot")

        if not all([model_id, dataset, validation_split, few_shot]):
            logger.debug(
                "Skipping record with missing fields: model_id=%s, dataset=%s",
                model_id,
                dataset,
            )
            return None

        return (model_id, dataset, str(validation_split), str(few_shot))
    except Exception as e:
        logger.debug("Failed to extract key from record: %s", e)
        return None


if __name__ == "__main__":
    main()
