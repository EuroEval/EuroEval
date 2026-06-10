"""Per-issue progress comment + results gist tracking.

While an evaluation is running, the queue maintains a single progress
comment on the issue (rewritten after every language) that links to a
private gist accumulating the JSONL results. :class:`ProgressState` carries
the gist id forward across the multiple comment refreshes so we update
one growing gist instead of creating a new gist on every refresh.
"""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.error
from dataclasses import dataclass, field

from .github_api import REPO, gh_request, patch_issue_body
from .queue_markers import GIST_MARKER_RE

logger = logging.getLogger(__name__)

PROGRESS_COMMENT_MARKER = "<!-- queue-progress -->"

# Matches the gist link rendered in the progress comment body so a fresh VM
# can pick the gist id back up from GitHub when its local results file is
# empty (e.g. continuing an evaluation orphaned by another VM).
GIST_URL_IN_COMMENT_RE = re.compile(r"gist\.github\.com/(?:[\w-]+/)?([0-9a-f]+)")


@dataclass
class ProgressState:
    """Per-issue state for the progress comment and the results gist.

    Carries the gist id forward across the multiple progress-comment
    updates that happen during a single issue's evaluation, so we update
    one growing gist instead of creating a new one on every refresh.

    Attributes:
        issue_number:
            The GitHub issue number this state belongs to.
        gist_id:
            The id of the results gist, set on first successful upload
            (or pre-populated when resuming a partial evaluation).
    """

    issue_number: int
    gist_id: str | None = None


@dataclass
class IncrementalGistUploader:
    """Uploads JSONL lines to a gist incrementally, avoiding duplicates.

    Tracks which lines have already been uploaded (by hash) and accumulates
    new lines. On each upload, PATCHes the gist with the full accumulated
    content so the gist always contains the complete result set.

    Attributes:
        state:
            The progress state holding the gist id.
        model_id:
            The Hugging Face model id used to name the gist file.
        uploaded_hashes:
            Set of SHA-256 hashes for lines already uploaded.
        accumulated_lines:
            All lines uploaded so far (in order).
    """

    state: ProgressState
    model_id: str
    uploaded_hashes: set[str] = field(default_factory=set, init=False)
    accumulated_lines: list[str] = field(default_factory=list, init=False)

    def seed_from_existing(self, existing_lines: list[str]) -> None:
        """Pre-populate with lines from a previous run.

        Args:
            existing_lines:
                Lines already in the results file (e.g. from partial_state).
        """
        for line in existing_lines:
            line_hash = hashlib.sha256(line.encode()).hexdigest()
            if line_hash not in self.uploaded_hashes:
                self.uploaded_hashes.add(line_hash)
                self.accumulated_lines.append(line)

    def add_new_lines(self, lines: list[str]) -> list[str]:
        """Add new lines to the accumulator, filtering out duplicates.

        Args:
            lines:
                Candidate lines to add (e.g. freshly read from the results
                file).

        Returns:
            The subset of lines that were actually new (not duplicates).
        """
        new_lines: list[str] = []
        for line in lines:
            line_hash = hashlib.sha256(line.encode()).hexdigest()
            if line_hash not in self.uploaded_hashes:
                self.uploaded_hashes.add(line_hash)
                self.accumulated_lines.append(line)
                new_lines.append(line)
        return new_lines

    def upload(self, issue_body: str | None = None) -> str | None:
        """Upload all accumulated lines to the gist.

        This PATCHes the existing gist (or creates a new one on first
        call) with the complete accumulated content.

        Args:
            issue_body (optional):
                The current issue body, used to store the gist marker on
                first upload. Defaults to None.

        Returns:
            The gist id if successful, or None on failure.
        """
        return upload_results_gist(
            state=self.state,
            model_id=self.model_id,
            lines=self.accumulated_lines,
            issue_body=issue_body,
        )


def find_partial_results_for_issue(number: int) -> dict | None:
    """Locate a usable progress comment + gist for ``number``.

    If the linked gist 404s, the stale progress comment is deleted so the
    issue starts from scratch.

    Args:
        number:
            The issue number to inspect.

    Returns:
        A dict with keys ``comment_id``, ``gist_id`` and ``lines`` when an
        existing progress comment links to a reachable gist, or None when
        no progress comment is present, when one exists but has not yet
        linked a gist, or when the linked gist could not be fetched.
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
    body = progress.get("body") or ""
    m = GIST_URL_IN_COMMENT_RE.search(body)
    if not m:
        return None
    gist_id = m.group(1)
    lines = fetch_gist_lines(gist_id=gist_id)
    if lines is None:
        delete_issue_comment(number=number, comment_id=comment_id)
        return None
    return {"comment_id": comment_id, "gist_id": gist_id, "lines": lines}


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
    lines: list[str],
    issue_body: str | None = None,
) -> int | None:
    """Create or PATCH the progress comment for the given issue.

    Args:
        state:
            The per-issue progress state. ``gist_id`` may be mutated on
            first successful upload.
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
        lines:
            The accumulated JSONL result lines.
        issue_body (optional):
            The current issue body, used to store the gist marker.
            Defaults to None.

    Returns:
        The id of the progress comment if it could be created or updated,
        otherwise None.
    """
    gist_url = None
    if lines:
        gist_id = upload_results_gist(
            state=state, model_id=model_id, lines=lines, issue_body=issue_body
        )
        if gist_id:
            gist_url = f"https://gist.github.com/{gist_id}"

    body = render_progress_comment(
        model_id=model_id,
        done=done,
        current=current,
        remaining=remaining,
        failed=failed,
        gist_url=gist_url,
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


def upload_results_gist(
    state: ProgressState, model_id: str, lines: list[str], issue_body: str | None = None
) -> str | None:
    """Upload JSONL results as a private gist, reusing the existing one.

    On first successful upload, ``state.gist_id`` is set so subsequent
    calls return the same id without creating a new gist. The gist id is
    also written into the issue body as an ``euroeval-results-gist``
    marker so it can be cleaned up after the issue closes.

    Args:
        state:
            The per-issue progress state. ``gist_id`` is updated in place.
        model_id:
            The Hugging Face model id used to name the gist file.
        lines:
            The JSONL result lines to upload.
        issue_body (optional):
            The current issue body, used to store the gist marker.
            Defaults to None.

    Returns:
        The gist id if a gist was created or already known, or None on
        failure.
    """
    filename = f"{model_id.replace('/', '_').replace('.', '_')}_results.jsonl"
    content = "\n".join(lines) + "\n" if lines else ""

    if state.gist_id:
        # Update the existing gist with the latest accumulated results.
        try:
            gh_request(
                path=f"/gists/{state.gist_id}",
                method="PATCH",
                body={"files": {filename: {"content": content}}},
            )
            logger.info(f"Updated results gist {state.gist_id} for {model_id!r}.")
            return state.gist_id
        except urllib.error.HTTPError as e:
            logger.warning(
                f"Could not update results gist {state.gist_id} for {model_id!r}: {e}"
            )
            return state.gist_id

    # Create a new gist (first language or no existing gist).
    try:
        resp = gh_request(
            path="/gists",
            method="POST",
            body={
                "description": f"EuroEval results for {model_id}",
                "files": {filename: {"content": content}},
                "public": False,
            },
        )
    except urllib.error.HTTPError as e:
        logger.warning(f"Could not create results gist for {model_id!r}: {e}")
        return None
    if not isinstance(resp, dict):
        return None
    gist_id = resp.get("id")
    if not isinstance(gist_id, str) or not gist_id:
        return None

    state.gist_id = gist_id
    if issue_body:
        cleaned = GIST_MARKER_RE.sub("", issue_body).rstrip()
        new_body = f"{cleaned}\n\n<!-- euroeval-results-gist: {gist_id} -->\n"
        try:
            patch_issue_body(number=state.issue_number, body=new_body)
            logger.info(f"Stored gist marker {gist_id} in issue body.")
        except urllib.error.HTTPError as e:
            logger.warning(f"Could not store gist marker in issue body: {e}")
    logger.info(f"Created results gist {gist_id} for {model_id!r}.")
    return gist_id


def render_progress_comment(
    model_id: str,
    done: list[str],
    current: str | None,
    remaining: list[str],
    failed: list[str],
    gist_url: str | None = None,
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
        gist_url (optional):
            URL to the results gist, or None if no gist has been created
            yet. Defaults to None.

    Returns:
        The markdown body of the progress comment, including the hidden
        ``queue-progress`` marker and a link to the results gist.
    """

    def fmt(langs: list[str]) -> str:
        return ", ".join(langs) if langs else "(none)"

    body = (
        f"{PROGRESS_COMMENT_MARKER}\n"
        f"**Evaluation progress for `{model_id}`**\n\n"
        f"- Completed: {fmt(done)}\n"
        f"- In progress: {current if current else '(none)'}\n"
        f"- Remaining: {fmt(remaining)}\n"
        f"- Failed: {fmt(failed)}\n"
    )
    if gist_url:
        body += f"\n\n[Benchmark results gist]({gist_url})\n"
    return body


def fetch_gist_lines(gist_id: str) -> list[str] | None:
    """Return the non-empty JSONL lines from a gist, or None if it 404s.

    Reads every file in the gist (typically there is just one ``*.jsonl``
    file) and concatenates their non-empty lines. Other failures are
    logged and also return None so callers can fall back to "no partial
    results".

    Args:
        gist_id:
            The GitHub gist id to fetch.

    Returns:
        The collected non-empty lines, or None if the gist could not be
        retrieved (including when it has been deleted).
    """
    try:
        resp = gh_request(path=f"/gists/{gist_id}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        logger.warning(f"Could not fetch gist {gist_id}: {e}")
        return None
    if not isinstance(resp, dict):
        return None
    files = resp.get("files") or {}
    if not isinstance(files, dict):
        return None
    lines: list[str] = []
    for file_data in files.values():
        if not isinstance(file_data, dict):
            continue
        content = file_data.get("content")
        if not isinstance(content, str):
            continue
        for raw_line in content.splitlines():
            if raw_line.strip():
                lines.append(raw_line)
    return lines


def delete_issue_comment(number: int, comment_id: int) -> None:
    """Delete a stale progress comment, logging on failure.

    Args:
        number:
            The issue number the comment belongs to (used for logging).
        comment_id:
            The id of the comment to delete.
    """
    try:
        gh_request(path=f"/repos/{REPO}/issues/comments/{comment_id}", method="DELETE")
        logger.info(f"#{number}: deleted stale progress comment {comment_id}.")
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not delete comment {comment_id}: {e}")
