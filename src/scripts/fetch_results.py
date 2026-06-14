"""Fetch and merge benchmark results from the Hugging Face bucket.

Syncs the raw-results bucket and merges all results into
euroeval_benchmark_results.jsonl, preserving existing local results.
"""

import json
import logging
from pathlib import Path

from euroeval.leaderboards.hf_mount import sync_bucket
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

    Deduplicates by (model_id, dataset, task, config) key.
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
                    key = (
                        rec.get("model_id"),
                        rec.get("dataset"),
                        rec.get("task"),
                        rec.get("config"),
                    )
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
                        key = (
                            rec.get("model_id"),
                            rec.get("dataset"),
                            rec.get("task"),
                            rec.get("config"),
                        )
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


if __name__ == "__main__":
    main()
