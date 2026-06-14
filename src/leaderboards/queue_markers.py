"""Issue-body marker manipulation for the queue scripts.

The queue uses HTML-comment markers in the issue body to track per-issue
state that needs to survive across runs and be readable by the frontend:

* ``<!-- vm-id: HOST-XXXX -->`` -- which VM currently owns the evaluation.

Note: The ``gated`` and ``evaluation-failed`` states are now tracked via
GitHub labels instead of body markers.
"""

from __future__ import annotations

import logging
import re
import urllib.error

from .github_api import fetch_issue_body, patch_issue_body, unassign_issue

logger = logging.getLogger(__name__)

VM_MARKER_RE = re.compile(r"<!--\s*vm-id:\s*([^\s>]+)\s*-->")


def set_vm_marker(number: int, vm_id: str) -> bool:
    """Stamp the issue body with the ``vm-id`` marker for ``vm_id``.

    The issue body is re-fetched so concurrent updates earlier in the same
    ``process_issue`` call (e.g. clearing a gated marker) are not clobbered.

    If another VM's marker is already present, the update is skipped to
    prevent race conditions where two VMs claim the same issue simultaneously.

    Args:
        number:
            The issue number to mark.
        vm_id:
            The VM identifier to record.

    Returns:
        True if the marker was set successfully; False if another VM already
        owns the issue (marker belongs to a different vm_id).
    """
    body = fetch_issue_body(number=number)
    m = VM_MARKER_RE.search(body)
    if m is not None and m.group(1) != vm_id:
        # Another VM already owns this issue
        return False
    cleaned = VM_MARKER_RE.sub("", body).rstrip()
    new_body = f"{cleaned}\n\n<!-- vm-id: {vm_id} -->\n"
    patch_issue_body(number=number, body=new_body)
    return True


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
