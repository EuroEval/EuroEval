"""Collect finished evaluation results from GitHub and regenerate leaderboards.

This script is meant to run on the maintainer's laptop. For each open
``model evaluation request`` issue that has been assigned (meaning it has
been picked up by the compute server), it:

1. Fetches the model's results JSONL from the HF bucket.
2. Concatenates all results into ``new_results.jsonl`` at the repo root.
3. Runs ``generate_leaderboards.py`` to merge the new results into the
   leaderboards.
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
HF_TOKEN            A Hugging Face token with write access to upload results.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

import click
from dotenv import load_dotenv
from huggingface_hub import BucketFile, HfApi
from huggingface_hub.errors import HfHubHTTPError

from leaderboards.backup import backup_results
from leaderboards.constants import (
    MODEL_REQUEST_LABEL,
    REPO,
    RESULTS_DIR,
    RESULTS_READY_LABEL,
)
from leaderboards.github_api import close_issue, comment_on_issue, gh_request
from leaderboards.queue_parsing import extract_model_id
from leaderboards.result_identity import (
    ResultIdentity,
    dedup_newer_record,
    identity_from_eee_record,
    identity_to_path,
    raise_on_collision,
    sanitise_model_dir_name,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("collect_evaluation_results")

REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_RESULTS_PATH = REPO_ROOT / "new_results.jsonl"

# Canonical HF bucket for storing evaluation results (public read access).
HF_RESULTS_BUCKET = "EuroEval/results"


@click.command()
@click.option(
    "--force/--no-force",
    "-f",
    default=False,
    show_default=True,
    help="Always regenerate leaderboards, even if no new results are found.",
)
def main(force: bool) -> None:
    """Harvest finished evaluations and regenerate leaderboards.

    Only issues with successfully harvested results are closed. Issues
    with the ``results-ready`` label may not yet have their results
    synced to the bucket, so the label alone is not sufficient.

    Args:
        force (optional):
            Whether to always regenerate leaderboards, even if no new results
            are found. Defaults to False.
    """
    check_required_env_vars()

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

    # When there are 0 open issues, continue with empty harvested results.
    # The bucket sync in upload_results_to_hf() handles incremental downloads.

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

    # Combine harvested results and manual results
    all_lines: list[str] = []
    for _, lines in harvested:
        all_lines.extend(lines)
    all_lines.extend(manual_lines)

    has_new_results = bool(all_lines)
    if not has_new_results:
        logger.info("Nothing to merge.")
        if not force:
            return
        logger.info("Forcing leaderboard regeneration despite no new results.")
    else:
        NEW_RESULTS_PATH.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

        harvested_count = sum(len(lines) for _, lines in harvested)
        logger.info(
            f"Wrote {len(all_lines)} line(s) to {NEW_RESULTS_PATH} "
            f"({harvested_count} harvested, {len(manual_lines)} manual)."
        )

        # Upload results to HF bucket BEFORE regenerating leaderboards
        # (leaderboard regeneration consumes/deletes new_results.jsonl)
        if upload_results_to_hf(new_results_path=NEW_RESULTS_PATH):
            logger.info("Results uploaded to Hugging Face bucket.")
        else:
            logger.error(
                "Failed to upload results to Hugging Face bucket. "
                "The new results are staged locally, but the bucket is now out "
                "of sync. Please run upload_results_to_hf() manually or check "
                "your Hugging Face credentials and re-run this script."
            )
            # Don't abort here -- leaderboards will still be correct because
            # load_raw_results() appends new_results.jsonl locally before deleting it.
            # But the bucket needs to be synced on the next successful run.

    # Regenerate leaderboards from merged results
    if not regenerate_leaderboards(force=force):
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
    backup_path = backup_results()
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


def check_required_env_vars() -> None:
    """Verify that the required tokens are set, exiting cleanly otherwise.

    ``HUGGINGFACE_API_KEY`` is accepted as an alias for ``HF_TOKEN`` (it is
    copied over when the ``leaderboards`` package is imported), so only
    ``HF_TOKEN`` is checked here. A missing ``HF_TOKEN`` would otherwise
    degrade silently into empty bucket reads and a failed upload.
    """
    missing = [var for var in ("GITHUB_TOKEN", "HF_TOKEN") if not os.environ.get(var)]
    if missing:
        logger.error(
            f"Missing required env var(s): {', '.join(missing)}. "
            "Set them (e.g. in a .env file) and re-run."
        )
        sys.exit(1)


def build_dedup_key(result: dict) -> ResultIdentity | None:
    """Build a deduplication key from an EEE result record.

    Delegates to the canonical ``identity_from_eee_record`` helper.

    Args:
        result:
            Parsed result dictionary in EEE format.

    Returns:
        Identity tuple ``(model_id, dataset, validation_split, few_shot)``,
        or None if required fields are missing.
    """
    try:
        return identity_from_eee_record(result)
    except (ValueError, KeyError) as e:
        logger.debug("Failed to extract identity from result: %s", e)
        return None


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
            "labels": f"{MODEL_REQUEST_LABEL},{RESULTS_READY_LABEL}",
            "per_page": "100",
        },
    )
    assert isinstance(issues, list)
    return [i for i in issues if "pull_request" not in i]


def find_results_for_issue(issue: dict) -> list[str] | None:
    """Fetch the model's JSON results from the HF bucket tree.

    Downloads all JSON files for the model's directory from the bucket
    and returns them as JSON strings.

    Args:
        issue:
            The issue object containing the model evaluation request.

    Returns:
        List of JSON strings for all results for this model,
        or None if no results exist yet.
    """
    number = issue["number"]
    title = issue.get("title") or ""
    body = issue.get("body")

    model_id = extract_model_id(title=title, body=body)
    if not model_id:
        logger.warning(f"#{number}: could not extract model_id from issue.")
        return None

    model_dir = sanitise_model_dir_name(model_id)

    try:
        api = HfApi()
        # List all files in the model's directory
        all_files = list(
            api.list_bucket_tree(bucket_id=HF_RESULTS_BUCKET, recursive=True)
        )
        model_files = [
            f
            for f in all_files
            if isinstance(f, BucketFile)
            and f.path.startswith(model_dir + "/")
            and f.path.endswith(".json")
        ]

        if not model_files:
            logger.info(f"#{number}: no results found for model {model_id}.")
            return None

        # Download all files for this model
        files_spec: list[tuple[str | BucketFile, str | Path]] = [
            (f, RESULTS_DIR / f.path) for f in model_files
        ]
        api.download_bucket_files(
            bucket_id=HF_RESULTS_BUCKET, files=files_spec, raise_on_missing_files=False
        )
    except Exception as e:
        logger.warning(f"#{number}: failed to download results from bucket: {e}")
        return None

    # Collect all JSON records
    results: list[str] = []
    for bucket_file in model_files:
        local_path = RESULTS_DIR / bucket_file.path
        if not local_path.exists():
            logger.warning(f"Downloaded file not found: {local_path}")
            continue
        try:
            content = local_path.read_text(encoding="utf-8")
            record = json.loads(content)  # validates and parses
            results.append(
                json.dumps(record, separators=(",", ":"))
            )  # compact to single line
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read {local_path}: {e}")

    if results:
        logger.info(f"#{number}: fetched {len(results)} result(s) from bucket.")
    return results


def _extract_identity_key(result: dict) -> ResultIdentity | None:
    """Extract the identity key from a result record.

    Args:
        result:
            The parsed result record.

    Returns:
        Identity tuple or None if extraction fails.
    """
    try:
        return identity_from_eee_record(result)
    except (ValueError, KeyError):
        return None


def upload_results_to_hf(new_results_path: Path) -> bool:
    """Upload results to Hugging Face bucket.

    Syncs the local results directory with the HF bucket (incremental download
    of changed files only), merges new results from the JSONL file,
    deduplicates by identity (newer records win), then syncs changed files
    back to the bucket.

    Handles both new harvested results and full bucket syncs when no new
    results are provided.

    Args:
        new_results_path:
            Path to the newly harvested results file (JSONL format).

    Returns:
        True if upload succeeded, False otherwise.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Sync existing results from EuroEval/results bucket
        logger.info(f"Syncing existing results from {HF_RESULTS_BUCKET}...")
        HfApi().sync_bucket(
            source=f"hf://buckets/{HF_RESULTS_BUCKET}/", dest=str(RESULTS_DIR)
        )
        logger.info("Downloaded existing results from bucket.")
    except HfHubHTTPError as e:
        logger.warning(f"Could not sync from bucket: {e}. Starting fresh.")

    # Load existing results as identity -> record dict
    existing: dict[ResultIdentity, dict] = {}
    for json_file in RESULTS_DIR.glob("*/*.json"):
        if not json_file.is_file():
            continue
        try:
            content = json_file.read_text(encoding="utf-8")
            record = json.loads(content)
            identity = _extract_identity_key(record)
            if identity:
                existing[identity] = record
        except (OSError, json.JSONDecodeError, ValueError):
            logger.debug(f"Skipping unreadable file {json_file}")

    # Process new results from JSONL file
    if not new_results_path.exists():
        logger.warning(f"Results file {new_results_path} does not exist.")
        return False

    new_lines = new_results_path.read_text(encoding="utf-8").splitlines()
    logger.info(f"Processing {len(new_lines):,} new result lines...")

    for line_number, line in enumerate(new_lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            identity = _extract_identity_key(record)
            if not identity:
                logger.debug(f"Skipping line {line_number}: no identity")
                continue

            # Dedup: keep newer record by euroeval_version, then retrieved_timestamp
            if identity in existing:
                existing[identity] = dedup_newer_record(existing[identity], record)
            else:
                existing[identity] = record
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON line: {line[:80]}...")

    # === VALIDATE PHASE: build path->identity map before any mutation ===
    path_to_identity: dict[Path, ResultIdentity] = {}
    for identity in existing:
        record_path = RESULTS_DIR / identity_to_path(identity)
        if record_path in path_to_identity:
            raise_on_collision(identity, path_to_identity[record_path])
        path_to_identity[record_path] = identity

    # === MUTATE PHASE: only after validation succeeds ===
    # Only rewrite records whose content actually changed. The initial bucket
    # download populated RESULTS_DIR with the current bucket state and we have
    # not modified it since, so comparing each desired record against the file
    # already on disk tells us exactly which records changed. Rewriting only the
    # changed files avoids touching every file's mtime and re-syncing the whole
    # tree on every run. We never drop identities (existing is only merged into,
    # never pruned), so there are no orphaned files to clear.
    records_written = 0
    records_unchanged = 0
    for identity, record in existing.items():
        record_path = RESULTS_DIR / identity_to_path(identity)
        desired = json.dumps(record, separators=(",", ":"))
        try:
            if (
                record_path.exists()
                and record_path.read_text(encoding="utf-8") == desired
            ):
                records_unchanged += 1
                continue
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(desired, encoding="utf-8")
            records_written += 1
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to write record for {identity}: {e}")

    # Remove stale ROOT-level RESULTS_DIR/*.jsonl artefacts (not repo-root files)
    for jsonl_file in RESULTS_DIR.glob("*.jsonl"):
        if jsonl_file.is_file():
            jsonl_file.unlink()
            logger.info(f"Removed stale artefact {jsonl_file}")

    if not existing:
        logger.warning("No valid results to upload.")
        return False

    if not records_written:
        logger.info(
            f"All {records_unchanged:,} records already up to date; nothing to sync."
        )
        return True

    logger.info(
        f"Wrote {records_written:,} changed record file(s) "
        f"({records_unchanged:,} unchanged) to {RESULTS_DIR}, syncing to bucket..."
    )

    # Sync to bucket
    try:
        HfApi().sync_bucket(
            source=str(RESULTS_DIR), dest=f"hf://buckets/{HF_RESULTS_BUCKET}/"
        )
        logger.info(f"Uploaded results to {HF_RESULTS_BUCKET}.")
    except HfHubHTTPError as e:
        logger.error(f"Failed to sync to bucket: {e}")
        return False

    return True


def regenerate_leaderboards(force: bool = False) -> bool:
    """Run the existing leaderboard-generation script.

    Args:
        force (optional):
            Whether to force leaderboard generation even if no updates are found.
            Defaults to False.

    Returns:
        True if the subprocess exited cleanly, otherwise False.
    """
    script_path = Path(__file__).resolve().parent / "generate_leaderboards.py"
    cmd = [sys.executable, str(script_path)]
    if force:
        cmd.append("--force")
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

    Leaderboards with <50 rows are skipped (not published) instead of failing
    the entire validation.

    Returns:
        True if validation completed (even if some leaderboards were skipped),
        False if critical errors occurred.
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

    skipped_count = 0
    valid_count = 0
    for csv_file in csv_files:
        try:
            with csv_file.open(mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if len(rows) < 50:
                    logger.warning(
                        f"{csv_file.name}: Only {len(rows)} rows (<50). "
                        "Skipping publication (too few results)."
                    )
                    csv_file.unlink()
                    skipped_count += 1
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
                        return False

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
                        return False

                logger.info(f"{csv_file.name}: {len(rows):,} rows OK")
                valid_count += 1
        except Exception as e:
            logger.error(f"{csv_file.name}: Failed to read: {e}")
            return False

    if skipped_count > 0:
        logger.info(
            f"Published {valid_count} leaderboard(s), skipped {skipped_count} "
            "(too few results)."
        )
    else:
        logger.info(f"All {valid_count} leaderboard CSVs passed sanity checks.")

    return valid_count > 0


def deploy_to_vercel() -> bool:
    """Build the frontend locally and ship it to Vercel as a prebuilt deploy.

    Using ``--prebuilt`` keeps the upload limited to ``.vercel/output/``
    so Vercel never sees the multi-hundred-MB ``.git`` packfile or local
    caches.

    Returns:
        True if both ``vercel build`` and ``vercel deploy --yes`` exit cleanly.
    """
    for cmd in (
        ["vercel", "build", "--prod", "--non-interactive"],
        ["vercel", "deploy", "--prebuilt", "--prod", "--yes", "--non-interactive"],
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
