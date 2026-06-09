"""GitHub REST API client and thin wrappers used by the queue scripts.

This module is the single home for talking to the GitHub API. Both the
queue-processor (server-side) and the results-collector (laptop-side)
scripts use the same client and the same set of thin helpers for
listing/labelling/assigning/closing issues.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import typing as t
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

REPO = "EuroEval/EuroEval"
LABEL = "model evaluation request"
FAILED_LABEL = "evaluation-failed"
GATED_LABEL = "Gated"
RESULTS_READY_LABEL = "results-ready"
TITLE_PREFIX = "[MODEL EVALUATION REQUEST]"

USER_AGENT = "euroeval-leaderboards"


def list_comments(number: int) -> list[dict[str, t.Any]]:
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


def assign_issue(number: int, assignee: str) -> None:
    """Assign the issue to ``assignee``.

    Args:
        number:
            The issue number to claim.
        assignee:
            The GitHub login to assign.
    """
    gh_request(
        path=f"/repos/{REPO}/issues/{number}/assignees",
        method="POST",
        body={"assignees": [assignee]},
    )


def unassign_issue(number: int, assignee: str) -> None:
    """Remove ``assignee`` so the issue returns to the unassigned pool.

    Args:
        number:
            The issue number to release.
        assignee:
            The GitHub login to remove.
    """
    gh_request(
        path=f"/repos/{REPO}/issues/{number}/assignees",
        method="DELETE",
        body={"assignees": [assignee]},
    )


def add_failed_label(number: int) -> None:
    """Attach the ``evaluation-failed`` label to an issue.

    Args:
        number:
            The issue number to label.
    """
    try:
        gh_request(
            path=f"/repos/{REPO}/issues/{number}/labels",
            method="POST",
            body={"labels": [FAILED_LABEL]},
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not add {FAILED_LABEL!r} label: {e}")


def remove_failed_label(number: int) -> None:
    """Remove the ``evaluation-failed`` label from an issue if present.

    Args:
        number:
            The issue number to unlabel.
    """
    try:
        gh_request(
            path=f"/repos/{REPO}/issues/{number}/labels/"
            + urllib.parse.quote(FAILED_LABEL),
            method="DELETE",
        )
    except urllib.error.HTTPError as e:
        if e.code != 404:
            logger.warning(f"#{number}: could not remove {FAILED_LABEL!r} label: {e}")


def add_gated_label(number: int) -> None:
    """Attach the ``Gated`` label to an issue.

    Args:
        number:
            The issue number to label.
    """
    gh_request(
        path=f"/repos/{REPO}/issues/{number}/labels",
        method="POST",
        body={"labels": [GATED_LABEL]},
    )


def remove_gated_label(number: int) -> None:
    """Remove the ``Gated`` label from an issue if present.

    Args:
        number:
            The issue number to unlabel.

    Raises:
        HTTPError:
            If the DELETE request fails for a reason other than label not present.
    """
    try:
        gh_request(
            path=f"/repos/{REPO}/issues/{number}/labels/{GATED_LABEL}", method="DELETE"
        )
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug(f"#{number}: label {GATED_LABEL!r} not present.")
        else:
            raise


def add_results_ready_label(number: int) -> None:
    """Attach the ``results-ready`` label to an issue.

    Args:
        number:
            The issue number to label.
    """
    try:
        gh_request(
            path=f"/repos/{REPO}/issues/{number}/labels",
            method="POST",
            body={"labels": [RESULTS_READY_LABEL]},
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not add {RESULTS_READY_LABEL!r} label: {e}")


def fetch_issue_body(number: int) -> str:
    """Return the current body of an issue, or empty string on failure.

    Args:
        number:
            The issue number to fetch.

    Returns:
        The issue body, or an empty string if the lookup failed.
    """
    try:
        current = gh_request(path=f"/repos/{REPO}/issues/{number}")
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not fetch issue body: {e}")
        return ""
    if isinstance(current, dict):
        return current.get("body") or ""
    return ""


def patch_issue_body(number: int, body: str) -> None:
    """Replace the body of an issue with ``body`` via a PATCH.

    Args:
        number:
            The issue number to update.
        body:
            The new markdown body.
    """
    gh_request(
        path=f"/repos/{REPO}/issues/{number}", method="PATCH", body={"body": body}
    )


def gh_request(
    path: str,
    *,
    method: str = "GET",
    body: dict[str, t.Any] | None = None,
    params: dict[str, t.Any] | None = None,
) -> dict[str, t.Any] | list[t.Any] | None:
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
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def _token() -> str:
    """Return the GitHub API token from the environment, or exit on failure.

    Returns:
        The value of the ``GITHUB_TOKEN`` environment variable.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN env var is required.")
        sys.exit(1)
    return token
