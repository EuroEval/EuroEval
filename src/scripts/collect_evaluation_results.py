"""Collect finished evaluation results from GitHub and regenerate leaderboards.

This script is meant to run on the maintainer's laptop. For each open
``model evaluation request`` issue that has been assigned (meaning it has
been picked up by the compute server), it:

1. Fetches the model's results JSONL from the HF bucket.
2. Concatenates all results into ``new_results.jsonl`` at the repo root.
3. Runs ``python -m scripts.generate_leaderboards`` to merge the new
   results into the leaderboards.
4. Builds the frontend and deploys it to Vercel as a prebuilt artifact
   (so Vercel's CLI never has to upload the >100 MB ``.git`` packfile).
5. Posts a comment and closes each processed issue.

To avoid race conditions, the script snapshots the list of results-ready
issues at the start (for logging purposes) and only closes issues with
successfully harvested results at the end (ignoring any new results-ready
issues that appeared during the run).

Required env vars
-----------------
GITHUB_TOKEN        A PAT with ``issues: write`` for the EuroEval repo.
HUGGINGFACE_API_KEY A Hugging Face token with write access to upload results.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
import tarfile
import urllib.error
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import BucketFile, HfApi
from huggingface_hub.errors import HfHubHTTPError

from euroeval.data_models import BenchmarkResult
from leaderboards.bucket_sync import create_backup
from leaderboards.github_api import (
    LABEL,
    REPO,
    RESULTS_READY_LABEL,
    close_issue,
    comment_on_issue,
    gh_request,
)
from leaderboards.paths import RAW_RESULTS_DIR, RESULTS_PATH
from leaderboards.queue_parsing import extract_model_id

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("collect_evaluation_results")

REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_RESULTS_PATH = REPO_ROOT / "new_results.jsonl"

# Canonical HF bucket for storing raw results (public read access).
HF_RAW_BUCKET = "hf://buckets/EuroEval/raw-results"


def _model_id_to_filename(model_id: str) -> str:
    """Convert a model ID to a safe filename.

    Args:
        model_id:
            The model identifier (e.g., "meta-llama/Llama-3-8B").

    Returns:
        A safe filename with slashes and dots replaced by underscores.
    """
    return model_id.replace("/", "_").replace(".", "_") + ".jsonl"


def build_dedup_key(result: dict) -> tuple[str, str, str, str] | None:
    """Build a deduplication key from a result record.

    Key consists of:
    - model_id: the model identifier
    - dataset: the dataset name
    - validation_split: whether eval used validation split
    - few_shot: whether eval used few-shot prompting

    Args:
        result:
            Parsed result dictionary (EEE or old format).

    Returns:
        Tuple key for deduplication, or None if required fields are missing.
    """
    try:
        model_id = result.get("model")
        dataset = result.get("dataset")
        if not model_id or not dataset:
            return None

        validation_split = result.get("validation_split", False)
        few_shot = result.get("few_shot", False)

        return (model_id, dataset, str(validation_split), str(few_shot))
    except Exception as e:
        logger.debug("Failed to extract dedup key from result: %s", e)
        return None


def convert_to_old_format(result_dict: dict) -> dict:
    """Convert an EEE-format result to the old format for backwards compatibility.

    Args:
        result_dict:
            Result dictionary in EEE format.

    Returns:
        Result dictionary in old format.
    """
    try:
        benchmark_result = BenchmarkResult.from_dict(result_dict)
        return benchmark_result.to_eee_dict()
    except Exception as e:
        logger.debug("Failed to convert result to old format: %s", e)
        return result_dict


def list_all_raw_result_files() -> list[BucketFile]:
    """List all .jsonl files in the raw-results bucket.

    Returns:
        List of BucketFile objects for .jsonl files.
    """
    api = HfApi()
    bucket_id = HF_RAW_BUCKET.replace("hf://buckets/", "")
    try:
        files = list(api.list_bucket_tree(bucket_id=bucket_id, recursive=True))
        return [
            f for f in files if isinstance(f, BucketFile) and f.path.endswith(".jsonl")
        ]
    except Exception as e:
        logger.error(f"Failed to list bucket files: {e}")
        return []


def load_existing_result_keys() -> set[tuple[str, str, str, str]]:
    """Load existing results and build a set of dedup keys.

    Loads results.tar.gz from RESULTS_PATH, extracts and parses
    results/results.jsonl, and builds a set of dedup keys.

    Returns:
        Set of dedup keys (model_id, dataset, validation_split, few_shot).
    """
    existing_keys: set[tuple[str, str, str, str]] = set()

    if not RESULTS_PATH.exists():
        logger.info("No existing results.tar.gz found.")
        return existing_keys

    try:
        with tarfile.open(RESULTS_PATH, "r:gz") as tar:
            results_member = tar.getmember("results/results.jsonl")
            results_file = tar.extractfile(results_member)
            if results_file is None:
                logger.warning("results/results.jsonl is empty in tar.gz.")
                return existing_keys

            for line in results_file:
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue
                try:
                    result = json.loads(line_str)
                    key = build_dedup_key(result)
                    if key is not None:
                        existing_keys.add(key)
                except json.JSONDecodeError:
                    logger.debug("Skipping invalid JSON line in existing results.")
    except Exception as e:
        logger.warning(f"Failed to load existing results: {e}")

    logger.info(f"Loaded {len(existing_keys)} existing result keys.")
    return existing_keys


def scan_bucket_for_results() -> list[str]:
    """Scan the raw-results bucket for new results not yet in results.tar.gz.

    Downloads all .jsonl files from the bucket, parses them, and collects
    results that are not already in the existing results.

    Returns:
        List of new result JSON strings (empty if none found).
    """
    logger.info("Scanning bucket for new results...")

    # Get all .jsonl files from the bucket
    all_bucket_files = list_all_raw_result_files()
    if not all_bucket_files:
        logger.info("No .jsonl files found in bucket.")
        return []

    logger.info(f"Found {len(all_bucket_files)} .jsonl file(s) in bucket.")

    # Download all files to RAW_RESULTS_DIR
    api = HfApi()
    bucket_id = HF_RAW_BUCKET.replace("hf://buckets/", "")

    files_spec: list[tuple[str | BucketFile, str | Path]] = [
        (bucket_file.path, RAW_RESULTS_DIR / bucket_file.path)
        for bucket_file in all_bucket_files
    ]

    try:
        api.download_bucket_files(
            bucket_id=bucket_id, files=files_spec, raise_on_missing_files=False
        )
        logger.info(f"Downloaded {len(files_spec)} file(s) to {RAW_RESULTS_DIR}.")
    except Exception as e:
        logger.error(f"Failed to download bucket files: {e}")
        return []

    # Load existing keys for deduplication
    existing_keys = load_existing_result_keys()

    # Parse all downloaded files and collect new results
    new_results: list[str] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    for bucket_file in all_bucket_files:
        local_path = RAW_RESULTS_DIR / bucket_file.path
        if not local_path.exists():
            logger.warning(f"Downloaded file not found: {local_path}")
            continue

        for line in local_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                result = json.loads(line)
                key = build_dedup_key(result)

                if key is None:
                    logger.debug(f"Skipping result with no dedup key: {line[:80]}...")
                    continue

                if key in existing_keys or key in seen_keys:
                    logger.debug(f"Skipping duplicate result: {key}")
                    continue

                # Convert to old format and add to new results
                converted = convert_to_old_format(result)
                new_results.append(json.dumps(converted, ensure_ascii=False))
                seen_keys.add(key)

            except json.JSONDecodeError as e:
                logger.debug(f"Skipping invalid JSON line: {e}")

    logger.info(f"Found {len(new_results)} new result(s) from bucket scan.")

    return new_results


def main() -> None:
    """Harvest finished evaluations and regenerate leaderboards.

    Only issues with successfully harvested results are closed. Issues
    with the ``results-ready`` label may not yet have their results
    synced to the bucket, so the label alone is not sufficient.
    """
    logger.info("Fetching open model evaluation request issues...")
    try:
        issues = list_open_request_issues()
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to list issues: {e}")
        sys.exit(1)
    logger.info(f"Found {len(issues)} open issue(s); scanning for results.")

    # Snapshot the issue numbers BEFORE any bucket operations to avoid
    # race conditions where new results-ready issues appear mid-run.
    snapshot_issue_numbers = {issue["number"] for issue in issues}
    logger.info(f"Snapshot: {len(snapshot_issue_numbers)} issue(s) to process.")

    harvested: list[tuple[int, list[str]]] = []
    for issue in issues:
        number = issue["number"]
        lines = find_results_for_issue(issue=issue)
        if not lines:
            logger.info(f"#{number}: no results in bucket yet -- skipping.")
            continue
        logger.info(f"#{number}: found {len(lines)} result line(s).")
        harvested.append((number, lines))

    # If no issues found, switch to bucket-scan mode
    bucket_scan_results: list[str] = []
    if len(issues) == 0:
        logger.info("No open issues found; using bucket-scan mode.")
        bucket_scan_results = scan_bucket_for_results()
        if not bucket_scan_results:
            logger.info("No new results found in bucket scan.")
            return
        logger.info(f"Bucket-scan mode found {len(bucket_scan_results)} new result(s).")

    # Load any manually added results from new_results.jsonl
    manual_lines: list[str] = []
    if NEW_RESULTS_PATH.exists():
        manual_lines = [
            line
            for line in NEW_RESULTS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if manual_lines:
            logger.info(f"Found {len(manual_lines)} manually added result line(s).")

    # Combine harvested results, bucket-scan results, and manual results
    all_lines: list[str] = []
    for _, lines in harvested:
        all_lines.extend(lines)
    all_lines.extend(bucket_scan_results)
    all_lines.extend(manual_lines)

    if not all_lines:
        logger.info("Nothing to merge.")
        return

    NEW_RESULTS_PATH.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    # Log which mode was used
    if bucket_scan_results:
        mode_str = f"bucket-scan ({len(bucket_scan_results)}), "
    else:
        mode_str = ""
    harvested_count = sum(len(lines) for _, lines in harvested)
    logger.info(
        f"Wrote {len(all_lines)} line(s) to {NEW_RESULTS_PATH} "
        f"({mode_str}{harvested_count} harvested, {len(manual_lines)} manual)."
    )

    # Upload results to HF bucket BEFORE regenerating leaderboards
    # (leaderboard regeneration consumes/deletes new_results.jsonl)
    if upload_results_to_hf(
        new_results_path=NEW_RESULTS_PATH, processed_path=RESULTS_PATH
    ):
        logger.info("Results uploaded to Hugging Face bucket.")
    else:
        logger.error(
            "Failed to upload results to Hugging Face bucket. "
            "The local archive (results.tar.gz) has been updated with the new results, "
            "but the bucket is now out of sync. Please run upload_results_to_hf() "
            "manually or check your Hugging Face credentials and re-run this script."
        )
        # Don't abort here -- leaderboards will still be correct because
        # load_raw_results() appends new_results.jsonl locally before deleting it.
        # But the bucket needs to be synced on the next successful run.

    if not regenerate_leaderboards():
        logger.error(
            "Aborting: not closing issues because leaderboard regeneration failed."
        )
        sys.exit(1)

    # Sanity check: verify leaderboards look sane before deploying
    if not verify_leaderboards():
        logger.error(
            "Aborting: leaderboard validation failed. "
            "Check the logs above, fix the issue, and redeploy manually."
        )
        sys.exit(1)

    if not deploy_to_vercel():
        logger.error("Aborting: not closing issues because the Vercel deploy failed.")
        sys.exit(1)

    # Create backup
    backup_path = create_backup()
    if backup_path:
        logger.info(f"Created backup at {backup_path}.")

    # Close ONLY the issues with successfully harvested results (not all snapshot
    # issues, as some may have the results-ready label but not yet synced to the
    # bucket).
    for number, _ in harvested:
        try:
            comment_on_issue(
                number=number, body="Results now live on the leaderboards!"
            )
            close_issue(number=number)
            logger.info(f"#{number}: closed.")
        except urllib.error.HTTPError as e:
            logger.error(f"#{number}: failed to close: {e}")


def list_open_request_issues() -> list[dict]:
    """Return open model-evaluation-request issues that have results ready.

    Returns:
        The list of open issues carrying both the queue label and the
        ``results-ready`` label, with pull requests filtered out.
    """
    issues = gh_request(
        path=f"/repos/{REPO}/issues",
        params={
            "state": "open",
            "labels": f"{LABEL},{RESULTS_READY_LABEL}",
            "per_page": "100",
        },
    )
    assert isinstance(issues, list)
    return [i for i in issues if "pull_request" not in i]


def find_results_for_issue(issue: dict) -> list[str] | None:
    """Fetch the model's JSONL results from the HF bucket.

    Args:
        issue:
            The issue object containing the model evaluation request.

    Returns:
        The non-empty lines of the model's JSONL file from the bucket,
        or None if the file does not exist yet.
    """
    number = issue["number"]
    title = issue.get("title") or ""
    body = issue.get("body")

    model_id = extract_model_id(title=title, body=body)
    if not model_id:
        logger.warning(f"#{number}: could not extract model_id from issue.")
        return None

    filename = _model_id_to_filename(model_id)
    local_path = RAW_RESULTS_DIR / filename

    try:
        api = HfApi()
        api.download_bucket_files(
            bucket_id=HF_RAW_BUCKET.replace("hf://buckets/", ""),
            files=[(filename, local_path)],
            raise_on_missing_files=False,
        )
    except Exception as e:
        logger.warning(f"#{number}: failed to download {filename} from bucket: {e}")
        return None

    if not local_path.exists():
        logger.info(f"#{number}: {filename} not found in bucket.")
        return None

    lines = [
        line
        for line in local_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if lines:
        logger.info(f"#{number}: fetched {len(lines)} line(s) from bucket.")
    return lines


def upload_results_to_hf(new_results_path: Path, processed_path: Path) -> bool:
    """Upload raw results to Hugging Face bucket.

    This function splits results into per-model files and syncs to raw-results
    bucket. Processed results are uploaded separately by result_processing.py as
    per-model JSONL.

    Args:
        new_results_path:
            Path to the newly harvested results.jsonl file.
        processed_path:
            Path to the processed results.tar.gz file (kept for backwards compatibility,
            but not uploaded to bucket - handled by result_processing.py).

    Returns:
        True if upload succeeded, False otherwise.
    """
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Sync existing results from raw-results bucket
        logger.info(f"Syncing existing results from {HF_RAW_BUCKET}...")
        HfApi().sync_bucket(source=HF_RAW_BUCKET + "/", dest=str(RAW_RESULTS_DIR))
        logger.info("Downloaded existing results from bucket.")
    except HfHubHTTPError as e:
        logger.warning(f"Could not sync from bucket: {e}. Starting fresh.")

    # Load existing results by model
    existing_by_model: dict[str, set[str]] = {}
    for model_file in RAW_RESULTS_DIR.glob("*.jsonl"):
        lines = {
            line
            for line in model_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        existing_by_model[model_file.name] = lines

    # Process new results and split into per-model files
    new_lines = new_results_path.read_text(encoding="utf-8").splitlines()
    logger.info(f"Processing {len(new_lines):,} new result lines...")

    for line in new_lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            model_id = data.get("model", "unknown")
            filename = _model_id_to_filename(model_id)
            model_file = RAW_RESULTS_DIR / filename
            # Append if not already present
            if line not in existing_by_model.get(filename, set()):
                with open(model_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON line: {line[:80]}...")

    # Sync updated results to raw-results bucket
    logger.info(f"Syncing results to {HF_RAW_BUCKET}...")
    try:
        HfApi().sync_bucket(source=str(RAW_RESULTS_DIR), dest=HF_RAW_BUCKET + "/")
        logger.info(f"Uploaded results to {HF_RAW_BUCKET}.")
    except HfHubHTTPError as e:
        logger.error(f"Failed to sync to bucket: {e}")
        return False

    # Note: Processed results are uploaded by result_processing.py, not here.
    # That script groups processed records by model and uploads per-model JSONL
    # files to HF_PROCESSED_BUCKET for consistency with raw-results format.

    return True


def regenerate_leaderboards() -> bool:
    """Run the existing leaderboard-generation script.

    Returns:
        True if the subprocess exited cleanly, otherwise False.
    """
    cmd = [sys.executable, "-m", "src.scripts.generate_leaderboards"]
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"generate_leaderboards failed (exit {e.returncode}).")
        return False


def verify_leaderboards() -> bool:
    """Verify that generated leaderboards look sane before deploying.

    Checks:
    - CSV files exist and are non-empty
    - Row count is reasonable (>100 models)
    - Required columns are present
    - No obvious data corruption (e.g., NaN in critical fields)

    Returns:
        True if all checks pass, False otherwise.
    """
    output_dir = REPO_ROOT / "src" / "frontend" / "csv"

    if not output_dir.exists():
        logger.error(f"Leaderboard output directory {output_dir} not found.")
        return False

    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in output directory.")
        return False

    logger.info(f"Found {len(csv_files)} leaderboard CSV(s).")

    all_passed = True
    for csv_file in csv_files:
        try:
            with csv_file.open(mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if len(rows) < 50:
                    logger.error(
                        f"{csv_file.name}: Only {len(rows)} rows (expected >=50). "
                        "Possible data loss?"
                    )
                    all_passed = False
                    continue

                if len(rows) < 100:
                    logger.warning(
                        f"{csv_file.name}: Only {len(rows)} rows (<100). "
                        "Expected for simplified/generative-only."
                    )

                # Check for critical columns
                # Some CSVs have HTML headers in row 0, actual headers in row 1
                # Re-read if HTML is present
                if rows and "rank" not in rows[0]:
                    # Re-read skipping HTML row
                    with csv_file.open(mode="r", encoding="utf-8") as f:
                        next(f)  # Skip HTML row
                        reader = csv.DictReader(f)  # Use row 1 as headers
                        rows = list(reader)

                if rows:
                    # Support both snake_case (simplified) and Title Case (full) formats
                    required_cols_snake = ["model", "mean_rank_score"]
                    required_cols_title = ["Model", "Rank score"]

                    missing_snake = [
                        col for col in required_cols_snake if col not in rows[0]
                    ]
                    missing_title = [
                        col for col in required_cols_title if col not in rows[0]
                    ]

                    # Must have at least one complete set
                    if missing_snake and missing_title:
                        logger.error(
                            f"{csv_file.name}: Missing required columns. "
                            f"Expected {required_cols_snake} or {required_cols_title}, "
                            f"got {list(rows[0].keys())[:5]}..."
                        )
                        all_passed = False
                        continue

                    # Use whichever format is present
                    model_col = "model" if not missing_snake else "Model"

                    # Check for NaN/None in critical fields
                    nan_count = sum(
                        1
                        for row in rows
                        if not row.get(model_col)
                        or row.get(model_col) in ("NaN", "None", "")
                    )
                    if nan_count > 0:
                        msg = f"{csv_file.name}: {nan_count} rows missing {model_col}."
                        logger.error(msg)
                        all_passed = False

                logger.info(f"{csv_file.name}: {len(rows):,} rows ✓")
        except Exception as e:
            logger.error(f"{csv_file.name}: Failed to read: {e}")
            all_passed = False

    if all_passed:
        logger.info("All leaderboard CSVs passed sanity checks.")
    else:
        logger.error("Leaderboard validation failed. Fix before deploying.")

    return all_passed


def deploy_to_vercel() -> bool:
    """Build the frontend locally and ship it to Vercel as a prebuilt deploy.

    Using ``--prebuilt`` keeps the upload limited to ``.vercel/output/``
    so Vercel never sees the multi-hundred-MB ``.git`` packfile or local
    caches.

    Returns:
        True if both ``vercel build`` and ``vercel deploy --yes`` exit cleanly.
    """
    for cmd in (
        ["vercel", "build", "--prod"],
        ["vercel", "deploy", "--prebuilt", "--prod", "--yes"],
    ):
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        except FileNotFoundError:
            logger.error(
                "`vercel` CLI not found on PATH. Install with `npm i -g vercel`."
            )
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"{cmd[0]} {cmd[1]} failed (exit {e.returncode}).")
            return False
    return True


if __name__ == "__main__":
    main()
