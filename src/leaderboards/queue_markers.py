"""Issue-body marker manipulation for the queue scripts.

The queue uses HTML-comment markers in the issue body to track per-issue
state that needs to survive across runs and be readable by the frontend:

* ``<!-- errored-on: vX.Y.Z -->`` -- the last EuroEval version that failed.
* ``<!-- gated-model -->`` -- the HF repo is gated and we lack read access.
* ``<!-- vm-id: HOST-XXXX -->`` -- which VM currently owns the evaluation.
* ``<!-- euroeval-results-gist: ID -->`` -- gist holding partial results.
"""

from __future__ import annotations

import logging
import re
import urllib.error

from .github_api import fetch_issue_body, patch_issue_body, unassign_issue

logger = logging.getLogger(__name__)

ERROR_MARKER_RE = re.compile(r"<!--\s*errored-on:\s*v([^\s>-]+)\s*-->")
GATED_MARKER_RE = re.compile(r"<!--\s*gated-model\s*-->")
VM_MARKER_RE = re.compile(r"<!--\s*vm-id:\s*([^\s>]+)\s*-->")
GIST_MARKER_RE = re.compile(r"<!--\s*euroeval-results-gist:\s*([^\s>]+)\s*-->")


def set_errored_marker(number: int, body: str | None, version: str) -> None:
    """Append/replace the ``errored-on`` marker in the issue body.

    Also strips any ``gated-model`` marker so the two states stay mutually
    exclusive.

    Args:
        number:
            The issue number to update.
        body:
            The current issue body, or None.
        version:
            The EuroEval version that produced the failure.
    """
    cleaned = ERROR_MARKER_RE.sub("", body or "").rstrip()
    cleaned = GATED_MARKER_RE.sub("", cleaned).rstrip()
    new_body = f"{cleaned}\n\n<!-- errored-on: v{version} -->\n"
    patch_issue_body(number=number, body=new_body)


def set_gated_marker(number: int, body: str | None) -> None:
    """Append the ``gated-model`` marker to the issue body.

    Also strips any ``errored-on`` marker so the two states stay mutually
    exclusive.

    Args:
        number:
            The issue number to update.
        body:
            The current issue body, or None.
    """
    cleaned = GATED_MARKER_RE.sub("", body or "").rstrip()
    cleaned = ERROR_MARKER_RE.sub("", cleaned).rstrip()
    new_body = f"{cleaned}\n\n<!-- gated-model -->\n"
    patch_issue_body(number=number, body=new_body)


def set_gated_with_errored_block(number: int, body: str | None, version: str) -> None:
    """Set both the ``gated-model`` and ``errored-on`` markers in one PATCH.

    Used when euroeval reports a gated repository despite ``model_info``
    succeeding: the script and the subprocess disagree on the token's
    download permission. The gated marker drives the UI status, while the
    errored marker prevents the script from re-running euroeval on every
    cron tick until the version bumps or access is reconfirmed.

    Args:
        number:
            The issue number to update.
        body:
            The current issue body, or None.
        version:
            The EuroEval version that observed the gated failure.
    """
    cleaned = GATED_MARKER_RE.sub("", body or "").rstrip()
    cleaned = ERROR_MARKER_RE.sub("", cleaned).rstrip()
    new_body = f"{cleaned}\n\n<!-- gated-model -->\n<!-- errored-on: v{version} -->\n"
    patch_issue_body(number=number, body=new_body)


def clear_gated_marker(number: int, body: str | None) -> None:
    """Remove the ``gated-model`` marker from the issue body.

    Args:
        number:
            The issue number to update.
        body:
            The current issue body, or None.
    """
    cleaned = GATED_MARKER_RE.sub("", body or "").rstrip() + "\n"
    patch_issue_body(number=number, body=cleaned)


def set_vm_marker(number: int, vm_id: str) -> None:
    """Stamp the issue body with the ``vm-id`` marker for ``vm_id``.

    The issue body is re-fetched so concurrent updates earlier in the same
    ``process_issue`` call (e.g. clearing a gated marker) are not clobbered.

    Args:
        number:
            The issue number to mark.
        vm_id:
            The VM identifier to record.
    """
    body = fetch_issue_body(number=number)
    cleaned = VM_MARKER_RE.sub("", body).rstrip()
    new_body = f"{cleaned}\n\n<!-- vm-id: {vm_id} -->\n"
    patch_issue_body(number=number, body=new_body)


def vm_marker_matches(number: int, vm_id: str) -> bool:
    """Return True if the issue body's vm-id marker is missing or matches ``vm_id``.

    A missing marker is treated as a match because the marker may have
    been cleared by an earlier step in the same release flow. A marker
    belonging to a different VM means another process has taken over the
    issue and we must not touch the assignment.

    Args:
        number:
            The issue number to check.
        vm_id:
            The VM identifier expected on the marker.

    Returns:
        True if the marker is absent or matches ``vm_id``; False if it
        belongs to a different VM.
    """
    body = fetch_issue_body(number=number)
    m = VM_MARKER_RE.search(body)
    return m is None or m.group(1) == vm_id


def release_issue_if_owned(number: int, vm_id: str, assignee: str) -> bool:
    """Clear our vm marker and unassign — only if this VM still owns the issue.

    Two queue processors sharing the same ``GITHUB_TOKEN`` owner cannot be
    distinguished by the issue assignee, so the body's ``vm-id`` marker is
    the source of truth for "who owns this issue right now". If the
    marker has been overwritten by another VM (e.g. because both VMs
    claimed the issue in a tight race between
    :func:`issue_is_still_claimable` and :func:`assign_issue`), this VM
    must not strip the assignee out from under the other VM's still-
    running evaluation.

    Args:
        number:
            The issue number to release.
        vm_id:
            The VM identifier this caller believes owns the issue.
        assignee:
            The GitHub login currently assigned to the issue.

    Returns:
        True if the issue was released; False if another VM has taken
        over and the assignment was left in place.
    """
    body = fetch_issue_body(number=number)
    m = VM_MARKER_RE.search(body)
    if m is not None and m.group(1) != vm_id:
        logger.info(
            f"#{number}: vm marker now belongs to {m.group(1)!r} (we are "
            f"{vm_id!r}); leaving assignee in place."
        )
        return False
    if m is not None:
        cleaned = VM_MARKER_RE.sub("", body, count=1).rstrip() + "\n"
        try:
            patch_issue_body(number=number, body=cleaned)
        except urllib.error.HTTPError as e:
            logger.warning(f"#{number}: could not clear vm marker: {e}")
    try:
        unassign_issue(number=number, assignee=assignee)
    except urllib.error.HTTPError as e:
        logger.warning(f"#{number}: could not unassign {assignee!r}: {e}")
    return True


def clear_vm_marker(number: int, vm_id: str) -> None:
    """Remove the ``vm-id`` marker from the body if it matches ``vm_id``.

    Re-fetches the body before patching so other markers set during the run
    are preserved. Markers belonging to other VMs are left untouched.

    Args:
        number:
            The issue number to clean.
        vm_id:
            The VM identifier that owns the marker to remove.
    """
    body = fetch_issue_body(number=number)
    m = VM_MARKER_RE.search(body)
    if not m or m.group(1) != vm_id:
        return
    cleaned = VM_MARKER_RE.sub("", body, count=1).rstrip() + "\n"
    patch_issue_body(number=number, body=cleaned)


def errored_marker_present(body: str | None) -> bool:
    """Return True if the body carries an ``errored-on`` marker.

    Args:
        body:
            The markdown body of the issue, or None.

    Returns:
        True if the marker is present, False otherwise.
    """
    return bool(body) and bool(ERROR_MARKER_RE.search(body))


def gated_marker_present(body: str | None) -> bool:
    """Return True if the body carries the ``gated-model`` marker.

    Args:
        body:
            The markdown body of the issue, or None.

    Returns:
        True if the marker is present, False otherwise.
    """
    return bool(body) and bool(GATED_MARKER_RE.search(body))
