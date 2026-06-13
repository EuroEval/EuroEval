"""Per-issue progress comment tracking for the evaluation queue.

While an evaluation is running, the queue maintains a single progress
comment on the issue showing which languages have been completed.
"""

from __future__ import annotations

import logging
import urllib.error
from dataclasses import dataclass

from .github_api import REPO, gh_request

logger = logging.getLogger(__name__)

PROGRESS_COMMENT_MARKER = "<!-- queue-progress -->"


@dataclass
class ProgressState:
    """Per-issue state for the progress comment.

    Attributes:
        issue_number:
            The GitHub issue number this state belongs to.
    """

    issue_number: int


def find_partial_results_for_issue(number: int) -> dict | None:
    """Locate a usable progress comment for ``number``.

    Args:
        number:
            The issue number to inspect.

    Returns:
        A dict with keys ``comment_id`` and ``lines`` when an
        existing progress comment is found, or None when no
        progress comment is present.
    """
    try:
        comments = gh_request(
            path=f"/repos/{REPO}/issues/{number}/comments", params={"per_page": "100"}
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not list comments: {e}")
        return None
    if not isinstance(comments, list):
        return None
    progress: dict | None = None
    for c in comments:
        if isinstance(c, dict) and PROGRESS_COMMENT_MARKER in (c.get("body") or ""):
            progress = c
            break
    if progress is None:
        return None
    comment_id = progress.get("id")
    if not isinstance(comment_id, int):
        return None
    # We no longer fetch lines from a gist; they come from the local results file.
    return {"comment_id": comment_id, "lines": []}


def find_progress_comment(number: int) -> int | None:
    """Return the id of the existing progress comment, or None.

    Args:
        number:
            The issue number to inspect.

    Returns:
        The comment id, or None if no progress comment is present.
    """
    try:
        comments = gh_request(
            path=f"/repos/{REPO}/issues/{number}/comments", params={"per_page": "100"}
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not list comments: {e}")
        return None
    if not isinstance(comments, list):
        return None
    for c in comments:
        if not isinstance(c, dict):
            continue
        if PROGRESS_COMMENT_MARKER in (c.get("body") or ""):
            cid = c.get("id")
            if isinstance(cid, int):
                return cid
    return None


def post_or_update_progress_comment(
    state: ProgressState,
    comment_id: int | None,
    model_id: str,
    done: list[str],
    current: str | None,
    remaining: list[str],
    failed: list[str],
) -> int | None:
    """Create or PATCH the progress comment for the given issue.

    Args:
        state:
            The per-issue progress state.
        comment_id:
            The id of an existing progress comment, or None to create one.
        model_id:
            The Hugging Face model id being evaluated.
        done:
            List of completed language codes.
        current:
            The currently processing language code, or None.
        remaining:
            List of remaining language codes.
        failed:
            List of failed language codes.

    Returns:
        The id of the progress comment if it could be created or updated,
        otherwise None.
    """
    body = render_progress_comment(
        model_id=model_id,
        done=done,
        current=current,
        remaining=remaining,
        failed=failed,
    )
    try:
        if comment_id is None:
            resp = gh_request(
                path=f"/repos/{REPO}/issues/{state.issue_number}/comments",
                method="POST",
                body={"body": body},
            )
            if isinstance(resp, dict):
                cid = resp.get("id")
                if isinstance(cid, int):
                    return cid
            return None
        gh_request(
            path=f"/repos/{REPO}/issues/comments/{comment_id}",
            method="PATCH",
            body={"body": body},
        )
        return comment_id
    except urllib.error.HTTPError as e:
        logger.warning(f"#{state.issue_number}: could not update progress comment: {e}")
        return comment_id


def render_progress_comment(
    model_id: str,
    done: list[str],
    current: str | None,
    remaining: list[str],
    failed: list[str],
) -> str:
    """Render the markdown body of the progress comment.

    Args:
        model_id:
            The Hugging Face model id.
        done:
            List of completed language codes.
        current:
            The currently processing language code, or None.
        remaining:
            List of remaining language codes.
        failed:
            List of failed language codes.

    Returns:
        The markdown body of the progress comment.
    """

    def fmt(langs: list[str]) -> str:
        return ", ".join(langs) if langs else "(none)"

    return (
        f"{PROGRESS_COMMENT_MARKER}\n"
        f"**Evaluation progress for `{model_id}`**\n\n"
        f"- Completed: {fmt(done)}\n"
        f"- In progress: {current if current else '(none)'}\n"
        f"- Remaining: {fmt(remaining)}\n"
        f"- Failed: {fmt(failed)}\n"
    )
