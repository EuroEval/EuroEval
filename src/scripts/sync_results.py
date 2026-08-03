"""Sync benchmark results bidirectionally with the Hugging Face bucket.

Downloads from the unified results bucket, merges all results into
euroeval_benchmark_results.jsonl, then uploads new local results back
to the bucket.
"""

import logging
from pathlib import Path

from leaderboards.bucket_sync import (
    download_missing_bucket_files,
    merge_results,
    upload_results_to_bucket,
)

logger = logging.getLogger(__name__)

RESULTS_FILE = Path("euroeval_benchmark_results.jsonl")


def main() -> None:
    """Main entry point.

    Downloads missing results from the Hugging Face bucket, merges them into a
    single JSONL file, and uploads any new local results back to the bucket.
    """
    logging.basicConfig(level=logging.INFO)
    download_missing_bucket_files()

    n_results = merge_results(results_file=RESULTS_FILE)
    if n_results > 0:
        logger.info(f"Merged {n_results:,} unique results into {RESULTS_FILE}")

    upload_results_to_bucket(RESULTS_FILE)


if __name__ == "__main__":
    main()
