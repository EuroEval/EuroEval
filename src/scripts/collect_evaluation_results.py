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
GITHUB_TOKEN   A PAT with ``issues: write`` for the EuroEval repo.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("collect_evaluation_results")

REPO = "EuroEval/EuroEval"
LABEL = "model evaluation request"
RESULTS_READY_LABEL = "results-ready"
REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_RESULTS_PATH = REPO_ROOT / "new_results.jsonl"

JSONL_FENCE_RE = re.compile(r"```jsonl\s*\n(.*?)\n```", re.DOTALL)

# Matches Markdown links to GitHub Gists (e.g. [Benchmark results gist](https://gist.github.com/abc123)).
GIST_LINK_RE = re.compile(r"https://gist\.github\.com/([a-zA-Z0-9]+)")


def main() -> None:
    """Harvest finished evaluations and regenerate leaderboards.

    Every issue returned by the listing is treated as complete: the queue
    processor only stamps ``results-ready`` once euroeval has finished every
    language (intentional skips included), so the label is authoritative.
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

    if not regenerate_leaderboards():
        logger.error(
            "Aborting: not closing issues because leaderboard regeneration failed."
        )
        sys.exit(1)

    if not deploy_to_vercel():
        logger.error("Aborting: not closing issues because the Vercel deploy failed.")
        sys.exit(1)

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
        A tuple of ``(lines, gist_id)``. ``lines`` holds the non-empty lines
        of the first jsonl fenced block or gist content, or None if no such
        content exists. ``gist_id`` is the gist id when results were fetched
        from a gist (so the caller can delete it after closing), otherwise
        None.

    Raises:
        urllib.error.HTTPError:
            If the GitHub API returns a non-404 error while fetching a gist
            referenced in a comment.
    """
    number = issue["number"]
    for comment in list_comments(number=number):
        # First, try inline jsonl block.
        block = extract_first_jsonl_block(text=comment.get("body") or "")
        if block:
            return [line for line in block.splitlines() if line.strip()], None

        # Then, try gist link.
        gist_id = extract_gist_id(comment.get("body") or "")
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
            If the GitHub API returns a 404 for the gist (so the caller can
            treat the reference as stale and clean it up).
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
            # GitHub's gist API truncates `content` for files larger than ~1 MB
            # and sets `truncated: true`. In that case the full file has to be
            # fetched from `raw_url`, otherwise downstream JSONL parsing fails
            # on the half-written final line.
            if first_file.get("truncated") and first_file.get("raw_url"):
                return _fetch_url_text(url=first_file["raw_url"])
            return first_file.get("content")
    return None


def _fetch_url_text(url: str) -> str | None:
    """Fetch a URL and return its body as text.

    Args:
        url:
            The URL to fetch.

    Returns:
        The decoded response body, or None if the request failed.
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": "euroeval-results-collector"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        logger.warning(f"Could not fetch gist raw url {url}: {e}")
        return None


def list_comments(number: int) -> list[dict]:
    """Return up to 100 comments for the issue with the given number.

    Args:
        number:
            The issue number whose comments should be fetched.

    Returns:
        The list of comment objects returned by the GitHub API.
    """
    comments = gh_request(
        path=f"/repos/{REPO}/issues/{number}/comments", params={"per_page": "100"}
    )
    assert isinstance(comments, list)
    return comments


def extract_first_jsonl_block(text: str) -> str | None:
    """Return the contents of the first ```jsonl ... ``` block in `text`.

    Args:
        text:
            The text to scan for a jsonl fenced block.

    Returns:
        The fenced block's inner text, stripped, or None if no block is found.
    """
    m = JSONL_FENCE_RE.search(text)
    return m.group(1).strip() if m else None


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


def deploy_to_vercel() -> bool:
    """Build the frontend locally and ship it to Vercel as a prebuilt deploy.

    Using ``--prebuilt`` keeps the upload limited to ``.vercel/output/`` so
    Vercel never sees the multi-hundred-MB ``.git`` packfile or local caches.

    Returns:
        True if both ``vercel build`` and ``vercel deploy`` exit cleanly.
    """
    for cmd in (
        ["vercel", "build", "--prod"],
        ["vercel", "deploy", "--prebuilt", "--prod"],
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


def comment_on_issue(number: int, body: str) -> None:
    """Post a comment on the issue with the given number.

    Args:
        number:
            The issue number to comment on.
        body:
            The markdown body of the new comment.
    """
    gh_request(
        path=f"/repos/{REPO}/issues/{number}/comments",
        method="POST",
        body={"body": body},
    )


def close_issue(number: int) -> None:
    """Mark the issue as closed with reason ``completed``.

    Args:
        number:
            The issue number to close.
    """
    gh_request(
        path=f"/repos/{REPO}/issues/{number}",
        method="PATCH",
        body={"state": "closed", "state_reason": "completed"},
    )


def gh_request(
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    params: dict | None = None,
) -> dict | list | None:
    """Call the GitHub REST API and return the parsed JSON body.

    Args:
        path:
            The API path, including the leading slash (e.g. ``/repos/...``).
        method (optional):
            The HTTP method to use. Defaults to "GET".
        body (optional):
            The JSON body to send, or None for no body. Defaults to None.
        params (optional):
            Query-string parameters, or None for none. Defaults to None.

    Returns:
        The parsed JSON response, or None if the body was empty.
    """
    url = f"https://api.github.com{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {_token()}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "euroeval-results-collector",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def _token() -> str:
    """Return the GitHub API token from the environment.

    Returns:
        The value of the ``GITHUB_TOKEN`` environment variable.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN env var is required.")
        sys.exit(1)
    return token


if __name__ == "__main__":
    main()
