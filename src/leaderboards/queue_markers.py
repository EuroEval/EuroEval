"""Strict GitHub ownership markers shared by the broker and local queue."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import re
import urllib.error

from .constants import VM_MARKER_RE
from .github_api import fetch_issue_body, patch_issue_body, unassign_issue

logger = logging.getLogger(__name__)

COMMUNITY_MARKER_VERSION = 1
COMMUNITY_PROTOCOL_VERSION = "volunteer-worker/v1"
COMMUNITY_MARKER_OWNER = "community"
COORDINATOR_MARKER_OWNER = "coordinator"
COMMUNITY_ACTIVE_SUBMISSION_STATES = frozenset(
    {"active", "pending", "running", "submitted"}
)
_COMMUNITY_SUBMISSION_STATES = COMMUNITY_ACTIVE_SUBMISSION_STATES | {
    "accepted",
    "rejected",
    # Read old markers during the migration, but never create these states.
    "completed",
    "released",
}
COMMUNITY_MARKER_RE = re.compile(
    r"<!--[ \t]*euroeval-volunteer-worker:v1[ \t]+(?P<payload>[^<]*?)-->"
)
_COMMUNITY_MARKER_CANDIDATE_RE = re.compile(r"<!--[ \t]*euroeval-volunteer-worker:v1")


@dataclasses.dataclass(frozen=True)
class CommunityMarker:
    """The broker's canonical ownership marker."""

    protocol_version: int
    owner: str
    submission: str
    leases: tuple[dict[str, str], ...] = ()
    submissions: tuple[dict[str, str], ...] = ()
    completed_languages: tuple[str, ...] = ()


def _expiry_active(value: str) -> bool:
    try:
        return dt.datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ) > dt.datetime.now(dt.UTC)
    except ValueError:
        return False


def parse_community_marker(body: str) -> CommunityMarker | None:
    """Parse exactly one canonical marker, failing closed on drift.

    Returns:
        The marker, or ``None`` when it is absent or invalid.
    """
    if len(_COMMUNITY_MARKER_CANDIDATE_RE.findall(body)) != 1:
        return None
    match = COMMUNITY_MARKER_RE.search(body)
    if match is None:
        return None
    try:
        payload = json.loads(match.group("payload").strip())
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(payload, dict)
        or not set(payload).issubset(
            {
                "protocol_version",
                "coordinator",
                "submission",
                "leases",
                "submissions",
                "completed_languages",
            }
        )
        or not {"protocol_version", "coordinator", "submission", "leases"}.issubset(
            payload
        )
    ):
        return None
    if (
        payload["protocol_version"] != COMMUNITY_PROTOCOL_VERSION
        or not isinstance(payload["coordinator"], str)
        or not isinstance(payload["submission"], str)
        or payload["submission"] not in _COMMUNITY_SUBMISSION_STATES
        or not isinstance(payload["leases"], list)
    ):
        return None
    submissions: list[dict[str, str]] = []
    raw_submissions = payload.get("submissions", [])
    if not isinstance(raw_submissions, list):
        return None
    for submission in raw_submissions:
        if (
            not isinstance(submission, dict)
            or not {
                "submission_id",
                "language",
                "manifest_path",
                "submitted_at",
            }.issubset(submission)
            or set(submission)
            - {
                "submission_id",
                "language",
                "manifest_path",
                "submitted_at",
                "contributor",
                "status",
            }
            or not all(
                isinstance(submission[key], str) and submission[key]
                for key in submission
            )
            or submission.get("status", "submitted")
            not in {"submitted", "accepted", "rejected"}
        ):
            return None
        submissions.append(submission)
    completed = payload.get("completed_languages", [])
    if not isinstance(completed, list) or not all(
        isinstance(item, str) and item for item in completed
    ):
        return None
    leases: list[dict[str, str]] = []
    for lease in payload["leases"]:
        if (
            not isinstance(lease, dict)
            or set(lease)
            != {"lease_id", "language", "worker", "contributor", "expires_at"}
            or not all(isinstance(lease[key], str) and lease[key] for key in lease)
            or not _expiry_active(lease["expires_at"])
        ):
            # Expired leases are retained for recovery but still validated.
            if (
                not isinstance(lease, dict)
                or set(lease)
                != {"lease_id", "language", "worker", "contributor", "expires_at"}
                or not all(isinstance(lease[key], str) and lease[key] for key in lease)
            ):
                return None
        try:
            dt.datetime.fromisoformat(lease["expires_at"].replace("Z", "+00:00"))
        except ValueError:
            return None
        leases.append(lease)
    if len({lease["lease_id"] for lease in leases}) != len(leases) or len(
        {lease["language"] for lease in leases}
    ) != len(leases):
        return None
    if len({item["submission_id"] for item in submissions}) != len(submissions) or len(
        {item["language"] for item in submissions}
    ) != len(submissions):
        return None
    return CommunityMarker(
        1,
        payload["coordinator"],
        payload["submission"],
        tuple(leases),
        tuple(submissions),
        tuple(completed),
    )


def issue_has_terminal_queue_submission(body: str) -> bool:
    """Return whether a valid marker records an accepted or rejected submission."""
    marker = parse_community_marker(body)
    return marker is not None and marker.submission in {"accepted", "rejected"}


def issue_has_active_queue_ownership(body: str) -> bool:
    """Return whether a valid, unexpired coordinator marker protects an issue."""
    marker = parse_community_marker(body)
    return (
        marker is not None
        and marker.submission in COMMUNITY_ACTIVE_SUBMISSION_STATES
        and (
            marker.submission == "submitted"
            or marker.submission in {"active", "pending", "running"}
            and (
                not marker.leases
                or any(_expiry_active(lease["expires_at"]) for lease in marker.leases)
            )
        )
    )


def append_community_marker(body: str, owner: str, submission: str) -> str:
    """Append a canonical marker for fixture and integration callers.

    Returns:
        The updated issue body.

    Raises:
        ValueError: If the owner or submission is unsupported.
    """
    if owner not in {COMMUNITY_MARKER_OWNER, COORDINATOR_MARKER_OWNER}:
        raise ValueError(f"Unsupported community marker owner: {owner!r}")
    if submission not in _COMMUNITY_SUBMISSION_STATES:
        raise ValueError(f"Unsupported community marker submission: {submission!r}")
    payload = {
        "protocol_version": COMMUNITY_PROTOCOL_VERSION,
        "coordinator": owner,
        "submission": submission,
        "leases": [],
    }
    encoded = json.dumps(payload, separators=(",", ":"))
    return f"{body.rstrip()}\n\n<!-- euroeval-volunteer-worker:v1 {encoded} -->\n"


def remove_community_marker(body: str) -> str:
    """Return ``body`` without its recognised canonical marker."""
    if parse_community_marker(body) is None:
        return body
    return COMMUNITY_MARKER_RE.sub("", body, count=1).rstrip() + "\n"


def clear_vm_marker(number: int, vm_id: str) -> None:
    """Remove this VM's marker while preserving active broker ownership."""
    body = fetch_issue_body(number=number)
    if issue_has_active_queue_ownership(body):
        logger.info(f"#{number}: coordinator ownership is active; keeping markers.")
        return
    match = VM_MARKER_RE.search(body)
    if match and match.group(1) == vm_id:
        patch_issue_body(
            number=number, body=VM_MARKER_RE.sub("", body, count=1).rstrip() + "\n"
        )


def release_issue_if_owned(number: int, vm_id: str, assignee: str) -> bool:
    """Release only after both marker clearing and unassignment succeed.

    Returns:
        Whether both GitHub mutations succeeded.
    """
    body = fetch_issue_body(number=number)
    if issue_has_active_queue_ownership(body):
        return False
    match = VM_MARKER_RE.search(body)
    if match and match.group(1) != vm_id:
        return False
    try:
        if match:
            patch_issue_body(
                number=number, body=VM_MARKER_RE.sub("", body, count=1).rstrip() + "\n"
            )
        unassign_issue(number=number, assignee=assignee)
    except urllib.error.HTTPError as error:
        logger.warning(f"#{number}: release failed: {error}")
        return False
    return True


def set_vm_marker(number: int, vm_id: str) -> bool:
    """Stamp an issue unless a valid broker lease owns it.

    Returns:
        Whether the marker was written.
    """
    body = fetch_issue_body(number=number)
    if issue_has_active_queue_ownership(body):
        return False
    match = VM_MARKER_RE.search(body)
    if match and match.group(1) != vm_id:
        return False
    if parse_community_marker(body) is not None:
        body = remove_community_marker(body)
    cleaned = VM_MARKER_RE.sub("", body).rstrip()
    patch_issue_body(number=number, body=f"{cleaned}\n\n<!-- vm-id: {vm_id} -->\n")
    return True


def vm_marker_matches(number: int, vm_id: str) -> bool:
    """Return whether the local VM marker is still safe to touch."""
    body = fetch_issue_body(number=number)
    if issue_has_active_queue_ownership(body):
        return False
    match = VM_MARKER_RE.search(body)
    return match is None or match.group(1) == vm_id
