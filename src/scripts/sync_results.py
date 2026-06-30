"""Sync benchmark results bidirectionally with the Hugging Face bucket.

Downloads from the unified results bucket, merges all results into
euroeval_benchmark_results.jsonl, then uploads new local results back
to the bucket.
"""

import json
import logging
from pathlib import Path

from euroeval.data_models import BenchmarkResult
from leaderboards.bucket_sync import sync_bucket, upload_results_to_bucket
from leaderboards.constants import RESULTS_DIR
from leaderboards.jsonl_io import parse_jsonl_lines

logger = logging.getLogger(__name__)

RESULTS_FILE = Path("euroeval_benchmark_results.jsonl")


def main() -> None:
    """Main entry point.

    Syncs results from the Hugging Face bucket, merges them into a single
    JSONL file, and uploads any new local results back to the bucket.
    """
    logging.basicConfig(level=logging.INFO)
    sync_bucket()

    n_results = merge_results()
    if n_results > 0:
        logger.info(f"Merged {n_results:,} unique results into {RESULTS_FILE}")

    upload_results_to_bucket(RESULTS_FILE)


def merge_results() -> int:
    """Merge all results into euroeval_benchmark_results.jsonl.

    Deduplicates by (model_id, dataset, validation_split, few_shot) key,
    which uniquely identifies an evaluation configuration.

    Existing local results are preserved; new results from the unified
    results directory are merged in.

    Returns:
        Number of unique results written.
    """
    existing: dict[tuple[str, str, str, str], str] = {}

    if RESULTS_FILE.exists():
        logger.info(f"Loading existing results from {RESULTS_FILE}...")
        records = parse_jsonl_lines(
            lines=RESULTS_FILE.read_text(encoding="utf-8").splitlines(),
            source=str(RESULTS_FILE),
        )
        for rec in records:
            result = BenchmarkResult.from_dict(config=rec)
            key = _extract_dedup_key(result=result)
            if key:
                # Re-serialize to ensure consistent formatting
                existing[key] = json.dumps(rec)
        logger.info(f"Found {len(existing):,} existing results")

    if RESULTS_DIR.exists():
        logger.info(f"Loading results from {RESULTS_DIR}...")
        bucket_count = 0
        for jsonl_file in sorted(RESULTS_DIR.glob("*.jsonl")):
            try:
                records = parse_jsonl_lines(
                    lines=jsonl_file.read_text(encoding="utf-8").splitlines(),
                    source=str(jsonl_file),
                )
                for rec in records:
                    try:
                        result = BenchmarkResult.from_dict(config=rec)
                        key = _extract_dedup_key(result=result)
                        if key:
                            # Re-serialize to ensure consistent formatting
                            existing[key] = json.dumps(rec)
                            bucket_count += 1
                    except Exception as e:
                        logger.debug(f"Skipping invalid record: {e}")
            except ValueError as e:
                logger.warning(f"Skipping malformed file {jsonl_file}: {e}")
        logger.info(f"Loaded {bucket_count:,} results from bucket")

    if not existing:
        logger.warning("No results found to merge")
        return 0

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        for line in sorted(existing.values()):
            f.write(line + "\n")

    return len(existing)


def _extract_dedup_key(result: BenchmarkResult) -> tuple[str, str, str, str] | None:
    """Extract deduplication key from a BenchmarkResult.

    The key is ``(model_id, dataset, validation_split, few_shot)``, which
    uniquely identifies an evaluation configuration.

    Args:
        result:
            Parsed benchmark result.

    Returns:
        Tuple key for deduplication, or None if required fields are missing.
    """
    if not result.model or not result.dataset:
        return None
    return (
        result.model,
        result.dataset,
        str(result.validation_split),
        str(result.few_shot),
    )


if __name__ == "__main__":
    main()
