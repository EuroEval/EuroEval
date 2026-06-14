"""Fetch and merge benchmark results from the Hugging Face bucket.

Syncs the raw-results bucket and merges all results into
euroeval_benchmark_results.jsonl, preserving existing local results.
"""

import json
import logging
import os
import subprocess
from pathlib import Path

from euroeval.leaderboards.paths import RAW_RESULTS_DIR

logger = logging.getLogger(__name__)

RESULTS_FILE = Path("euroeval_benchmark_results.jsonl")
HF_RAW_BUCKET = "EuroEval/raw-results"


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    sync_success = sync_bucket()
    if not sync_success:
        logger.warning("Continuing with local results only...")

    n_results = merge_results()
    if n_results > 0:
        logger.info("Merged %s unique results into %s", f"{n_results:,}", RESULTS_FILE)


def sync_bucket() -> bool:
    """Sync raw results from HF bucket to local directory.

    Reuses the same logic as leaderboards.hf_mount.sync_bucket(),
    but only syncs the raw bucket.

    Returns:
        True if sync succeeded, False otherwise.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Authenticate with `hf auth login` first.")
        return False

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Syncing hf://buckets/%s/ → %s...", HF_RAW_BUCKET, RAW_RESULTS_DIR)
    result = subprocess.run(
        ["hf", "sync", f"hf://buckets/{HF_RAW_BUCKET}/", str(RAW_RESULTS_DIR)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_TOKEN": hf_token},
    )
    if result.returncode != 0:
        logger.warning("hf sync failed: %s", result.stderr)
        return False

    logger.info("Synced: %s", result.stdout.strip())
    return True


def merge_results() -> int:
    """Merge all results into euroeval_benchmark_results.jsonl.

    Deduplicates by (model_id, dataset, task, config) key.
    Local results are preserved; bucket results fill in gaps.

    Reuses similar logic to leaderboards.result_loading._rebuild_results_tar_gz(),
    but outputs to euroeval_benchmark_results.jsonl instead of results.tar.gz.

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
