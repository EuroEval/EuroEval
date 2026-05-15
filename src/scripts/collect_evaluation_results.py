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
4. Closes the corresponding GitHub issues — which removes them from the
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
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(levelname)s ⋅ %(message)s"
)
logger = logging.getLogger("collect_evaluation_results")

REPO = "EuroEval/EuroEval"
LABEL = "model evaluation request"
REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_RESULTS_PATH = REPO_ROOT / "new_results.jsonl"

JSONL_FENCE_RE = re.compile(r"```jsonl\s*\n(.*?)\n```", re.DOTALL)


def _token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN env var is required.")
        sys.exit(1)
    return token


def gh_request(
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    params: dict | None = None,
) -> Any:  # noqa: ANN401
    """Call the GitHub REST API and return the parsed JSON body.

    Returns:
        The parsed JSON response, or ``None`` if the body was empty.
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


def list_assigned_open_issues() -> list[dict]:
    """Return open model-evaluation-request issues that have an assignee."""
    issues = gh_request(
        f"/repos/{REPO}/issues",
        params={"state": "open", "labels": LABEL, "per_page": "100", "assignee": "*"},
    )
    return [i for i in issues if "pull_request" not in i]


def list_comments(number: int) -> list[dict]:
    """Return up to 100 comments for the issue with the given number."""
    return gh_request(
        f"/repos/{REPO}/issues/{number}/comments", params={"per_page": "100"}
    )


def extract_first_jsonl_block(text: str) -> str | None:
    """Return the contents of the first ```jsonl ... ``` block in `text`."""
    m = JSONL_FENCE_RE.search(text)
    return m.group(1).strip() if m else None


def find_results_for_issue(number: int) -> list[str] | None:
    """Return the list of jsonl lines from the first jsonl block in comments."""
    for comment in list_comments(number):
        block = extract_first_jsonl_block(comment.get("body") or "")
        if block:
            return [line for line in block.splitlines() if line.strip()]
    return None


def close_issue(number: int) -> None:
    """Mark the issue as closed with reason ``completed``."""
    gh_request(
        f"/repos/{REPO}/issues/{number}",
        method="PATCH",
        body={"state": "closed", "state_reason": "completed"},
    )


def comment_on_issue(number: int, body: str) -> None:
    """Post a comment on the issue with the given number."""
    gh_request(
        f"/repos/{REPO}/issues/{number}/comments", method="POST", body={"body": body}
    )


def regenerate_leaderboards() -> bool:
    """Run the existing leaderboard-generation script.

    Returns:
        ``True`` if the subprocess exited cleanly, otherwise ``False``.
    """
    cmd = [sys.executable, "-m", "scripts.generate_leaderboards"]
    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT / "src")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("generate_leaderboards failed (exit %s).", e.returncode)
        return False


def main() -> None:
    """Harvest finished evaluations, regenerate leaderboards, close issues."""
    try:
        issues = list_assigned_open_issues()
    except urllib.error.HTTPError as e:
        logger.error("Failed to list issues: %s", e)
        sys.exit(1)

    completed: list[tuple[int, list[str]]] = []
    for issue in issues:
        number = issue["number"]
        lines = find_results_for_issue(number)
        if not lines:
            logger.info("#%s: no jsonl block in comments yet — skipping.", number)
            continue
        logger.info("#%s: found %d result line(s).", number, len(lines))
        completed.append((number, lines))

    if not completed:
        logger.info("Nothing to merge.")
        return

    all_lines: list[str] = []
    for _, lines in completed:
        all_lines.extend(lines)

    NEW_RESULTS_PATH.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    logger.info("Wrote %d line(s) to %s.", len(all_lines), NEW_RESULTS_PATH)

    if not regenerate_leaderboards():
        logger.error(
            "Aborting: not closing issues because leaderboard regeneration failed."
        )
        sys.exit(1)

    for number, _ in completed:
        try:
            comment_on_issue(
                number,
                "Results merged into the leaderboards. Thanks for the submission! 🎉",
            )
            close_issue(number)
            logger.info("#%s: closed.", number)
        except urllib.error.HTTPError as e:
            logger.error("#%s: failed to close: %s", number, e)


if __name__ == "__main__":
    main()
