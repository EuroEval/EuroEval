"""Collect finished evaluation results from GitHub and regenerate leaderboards.

This script is meant to run on the maintainer's laptop. For each open
``model evaluation request`` issue that has been assigned (meaning it has
been picked up by the compute server) and has at least one comment with a
``jsonl`` code fence, it:

1. Extracts the first ``jsonl`` fenced block from the issue comments.
2. Concatenates all those blocks into ``new_results.jsonl`` at the repo
   root, overwriting any previous file.
3. Runs ``python -m scripts.generate_leaderboards`` to merge the new
   results into the leaderboards.
4. Builds the frontend and deploys it to Vercel as a prebuilt artifact
   (so Vercel's CLI never has to upload the >100 MB ``.git`` packfile).
5. Closes the corresponding GitHub issues -- which removes them from the
   queue in the frontend.

Required env vars
-----------------
GITHUB_TOKEN        A PAT with ``issues: write`` for the EuroEval repo.
HUGGINGFACE_API_KEY A Hugging Face token with write access to upload results.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from leaderboards.github_api import (
    LABEL,
    REPO,
    RESULTS_READY_LABEL,
    USER_AGENT,
    close_issue,
    comment_on_issue,
    gh_request,
    list_comments,
)
from leaderboards.hf_mount import create_backup
from leaderboards.paths import RAW_RESULTS_DIR, RESULTS_PATH

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


JSONL_FENCE_RE = re.compile(r"```jsonl\s*\n(.*?)\n```", re.DOTALL)

# Matches Markdown links to GitHub Gists (e.g.
# [Benchmark results gist](https://gist.github.com/abc123)).
GIST_LINK_RE = re.compile(r"https://gist\.github\.com/([a-zA-Z0-9]+)")


def main() -> None:
    """Harvest finished evaluations and regenerate leaderboards.

    Every issue returned by the listing is treated as complete: the
    queue processor only stamps ``results-ready`` once euroeval has
    finished every language (intentional skips included), so the label
    is authoritative.
    """
    logger.info("Fetching open model evaluation request issues...")
    try:
        issues = list_open_request_issues()
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to list issues: {e}")
        sys.exit(1)
    logger.info(f"Found {len(issues)} open issue(s); scanning for results.")

    harvested: list[tuple[int, list[str], str | None]] = []
    for issue in issues:
        number = issue["number"]
        lines, gist_id = find_results_for_issue(issue=issue)
        if not lines:
            logger.info(f"#{number}: no jsonl block in comments yet -- skipping.")
            continue
        logger.info(f"#{number}: found {len(lines)} result line(s).")
        harvested.append((number, lines, gist_id))

    if not harvested:
        logger.info("Nothing to merge.")
        return

    all_lines: list[str] = []
    for _, lines, _ in harvested:
        all_lines.extend(lines)

    NEW_RESULTS_PATH.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {len(all_lines)} line(s) to {NEW_RESULTS_PATH}.")

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

    # Create backup if using hf-mount
    backup_path = create_backup()
    if backup_path:
        logger.info(f"Created backup at {backup_path}.")

    for number, _, gist_id in harvested:
        try:
            comment_on_issue(
                number=number, body="Results now live on the leaderboards 🎉"
            )
            close_issue(number=number)
            logger.info(f"#{number}: closed.")
        except urllib.error.HTTPError as e:
            logger.error(f"#{number}: failed to close: {e}")
            continue
        if gist_id:
            try:
                gh_request(path=f"/gists/{gist_id}", method="DELETE")
                logger.info(f"#{number}: deleted results gist {gist_id}.")
            except urllib.error.HTTPError as e:
                logger.warning(f"#{number}: could not delete gist {gist_id}: {e}")


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


def find_results_for_issue(issue: dict) -> tuple[list[str] | None, str | None]:
    """Return jsonl lines and gist id from the first jsonl block or gist link.

    Args:
        issue:
            The issue object whose comments should be inspected.

    Returns:
        A tuple of ``(lines, gist_id)``. ``lines`` holds the non-empty
        lines of the first jsonl fenced block or gist content, or None
        if no such content exists. ``gist_id`` is the gist id when
        results were fetched from a gist (so the caller can delete it
        after closing), otherwise None.

    Raises:
        urllib.error.HTTPError:
            If the GitHub API returns a non-404 error while fetching a
            gist referenced in a comment.
    """
    number = issue["number"]
    for comment in list_comments(number=number):
        block = extract_first_jsonl_block(text=comment.get("body") or "")
        if block:
            return [line for line in block.splitlines() if line.strip()], None

        gist_id = extract_gist_id(text=comment.get("body") or "")
        if gist_id:
            try:
                gist_content = fetch_gist_content(gist_id=gist_id)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    logger.warning(
                        f"#{number}: gist {gist_id} is missing (404); "
                        "cleaning up stale comment."
                    )
                    handle_stale_gist_comment(issue=issue, comment=comment)
                    continue
                raise
            if gist_content:
                return [
                    line for line in gist_content.splitlines() if line.strip()
                ], gist_id
    return None, None


def handle_stale_gist_comment(issue: dict, comment: dict) -> None:
    """Clean up an issue whose gist link no longer resolves.

    Deletes the offending comment, unassigns any current assignees, and
    removes the ``results-ready`` label if present. Failures on any
    individual step are logged but do not abort the others.

    Args:
        issue:
            The issue object the stale comment belongs to.
        comment:
            The comment object whose gist link 404'd.
    """
    number = issue["number"]
    comment_id = comment.get("id")
    if comment_id is not None:
        try:
            gh_request(
                path=f"/repos/{REPO}/issues/comments/{comment_id}", method="DELETE"
            )
            logger.info(f"#{number}: deleted stale gist comment {comment_id}.")
        except urllib.error.HTTPError as e:
            logger.warning(f"#{number}: could not delete comment {comment_id}: {e}")

    assignees = [a["login"] for a in issue.get("assignees") or [] if a.get("login")]
    if assignees:
        try:
            gh_request(
                path=f"/repos/{REPO}/issues/{number}/assignees",
                method="DELETE",
                body={"assignees": assignees},
            )
            logger.info(
                f"#{number}: unassigned {', '.join(assignees)} after stale gist."
            )
        except urllib.error.HTTPError as e:
            logger.warning(f"#{number}: could not unassign {assignees}: {e}")

    has_results_ready = any(
        (lbl.get("name") if isinstance(lbl, dict) else lbl) == "results-ready"
        for lbl in issue.get("labels") or []
    )
    if has_results_ready:
        label = urllib.parse.quote("results-ready", safe="")
        try:
            gh_request(
                path=f"/repos/{REPO}/issues/{number}/labels/{label}", method="DELETE"
            )
            logger.info(f"#{number}: removed `results-ready` label after stale gist.")
        except urllib.error.HTTPError as e:
            logger.warning(f"#{number}: could not remove `results-ready` label: {e}")


def extract_first_jsonl_block(text: str) -> str | None:
    """Return the contents of the first ``jsonl`` fenced block in ``text``.

    Args:
        text:
            The text to scan for a jsonl fenced block.

    Returns:
        The fenced block's inner text, stripped, or None if no block is
        found.
    """
    m = JSONL_FENCE_RE.search(text)
    return m.group(1).strip() if m else None


def extract_gist_id(text: str) -> str | None:
    """Return the gist ID from the first gist link in ``text``.

    Args:
        text:
            The text to scan for a gist link.

    Returns:
        The gist ID if found, or None.
    """
    m = GIST_LINK_RE.search(text)
    return m.group(1) if m else None


def fetch_gist_content(gist_id: str) -> str | None:
    """Fetch the content of the first file in a GitHub Gist.

    Args:
        gist_id:
            The GitHub Gist ID.

    Returns:
        The content of the first file in the gist, or None on failure.

    Raises:
        urllib.error.HTTPError:
            If the GitHub API returns a 404 for the gist (so the caller
            can treat the reference as stale and clean it up).
    """
    try:
        resp = gh_request(path=f"/gists/{gist_id}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise
        logger.warning(f"Could not fetch gist {gist_id}: {e}")
        return None
    if isinstance(resp, dict) and "files" in resp:
        first_file = next(iter(resp["files"].values()))
        if isinstance(first_file, dict):
            # GitHub's gist API truncates `content` for files larger than ~1
            # MB and sets `truncated: true`. In that case the full file has
            # to be fetched from `raw_url`, otherwise downstream JSONL
            # parsing fails on the half-written final line.
            if first_file.get("truncated") and first_file.get("raw_url"):
                return _fetch_url_text(url=first_file["raw_url"])
            return first_file.get("content")
    return None


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
    cmd = [sys.executable, "-m", "scripts.generate_leaderboards"]
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT / "src")
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

                if len(rows) < 100:
                    logger.error(
                        f"{csv_file.name}: Only {len(rows)} rows (expected >100). "
                        "Possible data loss?"
                    )
                    all_passed = False
                    continue

                # Check for critical columns
                # Some CSVs have HTML headers in row 0, actual headers in row 1
                header_row = rows[0]
                if "rank" not in header_row and len(rows) > 1 and "rank" in rows[1]:
                    header_row = rows[1]
                    rows = rows[1:]  # Use data rows only

                if rows:
                    required_cols = ["model", "mean_rank_score"]
                    missing = [col for col in required_cols if col not in header_row]
                    if missing:
                        logger.error(
                            f"{csv_file.name}: Missing required columns: {missing}"
                        )
                        all_passed = False
                        continue

                    # Check for NaN/None in critical fields
                    nan_count = sum(
                        1
                        for row in rows
                        if not row.get("model")
                        or row.get("model") in ("NaN", "None", "")
                    )
                    if nan_count > 0:
                        msg = f"{csv_file.name}: {nan_count} rows with missing model."
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


def _fetch_url_text(url: str) -> str | None:
    """Fetch a URL and return its body as text.

    Args:
        url:
            The URL to fetch.

    Returns:
        The decoded response body, or None if the request failed.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        logger.warning(f"Could not fetch gist raw url {url}: {e}")
        return None


if __name__ == "__main__":
    main()
