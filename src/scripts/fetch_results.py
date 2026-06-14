"""Fetch and merge benchmark results from the Hugging Face bucket.

Syncs the raw-results bucket and merges all results into
euroeval_benchmark_results.jsonl, preserving existing local results.
"""

import json
import os
import subprocess
from pathlib import Path

RESULTS_FILE = Path("euroeval_benchmark_results.jsonl")
RAW_DIR = Path("results/raw")
HF_BUCKET = "hf://buckets/EuroEval/raw-results/"


def sync_bucket() -> bool:
    """Sync raw results from HF bucket to local directory.

    Returns:
        True if sync succeeded, False otherwise.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠ HF_TOKEN not set. Authenticate with `hf auth login` first.")
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Syncing {HF_BUCKET} → {RAW_DIR}...")
    result = subprocess.run(
        ["hf", "sync", HF_BUCKET, str(RAW_DIR)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_TOKEN": hf_token},
    )
    if result.returncode != 0:
        print(f"⚠ hf sync failed: {result.stderr}")
        return False

    print(f"✓ Synced: {result.stdout.strip()}")
    return True


def merge_results() -> int:
    """Merge all results into euroeval_benchmark_results.jsonl.

    Deduplicates by (model_id, dataset, task, config) key.
    Local results are preserved; bucket results fill in gaps.

    Returns:
        Number of unique results written.
    """
    existing: dict[tuple, str] = {}

    # Load existing local results
    if RESULTS_FILE.exists():
        print(f"Loading existing results from {RESULTS_FILE}...")
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
        print(f"  Found {len(existing):,} existing results")

    # Load results from raw bucket
    if RAW_DIR.exists():
        print(f"Loading results from {RAW_DIR}...")
        bucket_count = 0
        for jsonl_file in sorted(RAW_DIR.glob("*.jsonl")):
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
        print(f"  Loaded {bucket_count:,} results from bucket")

    if not existing:
        print("⚠ No results found to merge")
        return 0

    # Write merged results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w") as f:
        for line in sorted(existing.values()):
            f.write(line + "\n")

    return len(existing)


def main() -> None:
    """Main entry point."""
    sync_success = sync_bucket()
    if not sync_success:
        print("⚠ Continuing with local results only...")

    n_results = merge_results()
    if n_results > 0:
        print(f"✓ Merged {n_results:,} unique results into {RESULTS_FILE}")


if __name__ == "__main__":
    main()
