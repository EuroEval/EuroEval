"""Strict GitHub ownership markers shared by the broker and local queue."""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
import urllib.error

from .constants import VM_MARKER_RE
from .github_api import fetch_issue_body, patch_issue_body, unassign_issue

logger = logging.getLogger(__name__)

COMMUNITY_MARKER_VERSION = 1
COMMUNITY_PROTOCOL_VERSION = "volunteer-worker/v1"
COMMUNITY_MARKER_OWNER = "community"
COORDINATOR_MARKER_OWNER = "coordinator"
COMMUNITY_ACTIVE_SUBMISSION_STATES = frozenset({"active", "submitted"})
_COMMUNITY_SUBMISSION_STATES = COMMUNITY_ACTIVE_SUBMISSION_STATES | {
    "accepted",
    "rejected",
}
COMMUNITY_MARKER_RE = re.compile(
    r"<!--[ \t]*euroeval-volunteer-worker:v1[ \t]+(?P<payload>[^<]*?)-->"
)
_COMMUNITY_MARKER_CANDIDATE_RE = re.compile(r"<!--[ \t]*euroeval-volunteer-worker:v1")


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


def clear_vm_marker(number: int, vm_id: str) -> None:
    """Remove this VM's marker while preserving active broker ownership."""
    body = fetch_issue_body(number=number)
    if (
        _COMMUNITY_MARKER_CANDIDATE_RE.search(body)
        and _trusted_issue_marker(number, body) is None
    ):
        return
    if issue_has_active_queue_ownership(body):
        logger.info(f"#{number}: coordinator ownership is active; keeping markers.")
        return
    match = VM_MARKER_RE.search(body)
    if match and match.group(1) == vm_id:
        patch_issue_body(
            number=number, body=VM_MARKER_RE.sub("", body, count=1).rstrip() + "\n"
        )


@dataclasses.dataclass(frozen=True)
class CommunityMarker:
    """The broker's canonical ownership marker."""

    protocol_version: int
    owner: str
    submission: str
    leases: tuple[dict[str, str], ...] = ()
    submissions: tuple[dict[str, object], ...] = ()
    completed_languages: tuple[str, ...] = ()
    signature: str | None = None


def _trusted_issue_marker(number: int, body: str) -> CommunityMarker | None:
    secret = os.environ.get("VOLUNTEER_MARKER_SECRET")
    marker = parse_community_marker(
        body, issue_number=number, secret=secret, require_signature=bool(secret)
    )
    if marker is not None and marker.submission == "accepted" and not secret:
        return None
    return marker


def parse_community_marker(
    body: str,
    *,
    issue_number: int | None = None,
    secret: str | None = None,
    require_signature: bool = False,
) -> CommunityMarker | None:
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
                "signature",
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
    submissions: list[dict[str, object]] = []
    raw_submissions = payload.get("submissions", [])
    if not isinstance(raw_submissions, list):
        return None
    for submission in raw_submissions:
        if (
            not isinstance(submission, dict)
            or set(submission)
            != {
                "submission_id",
                "language",
                "manifest_path",
                "submitted_at",
                "verified_contributor",
                "result_count",
                "status",
            }
            or not all(
                isinstance(submission[key], str) and submission[key]
                for key in {
                    "submission_id",
                    "language",
                    "manifest_path",
                    "submitted_at",
                    "verified_contributor",
                    "status",
                }
            )
            or not isinstance(submission["result_count"], int)
            or isinstance(submission["result_count"], bool)
            or submission["result_count"] <= 0
            or submission["status"] not in {"submitted", "accepted", "rejected"}
        ):
            return None
        submissions.append(submission)
    signature = payload.get("signature")
    if signature is not None and (not isinstance(signature, str) or not signature):
        return None
    if require_signature and (
        issue_number is None or not secret or not isinstance(signature, str)
    ):
        return None
    if signature is not None and issue_number is not None and secret:
        unsigned = dict(payload)
        unsigned.pop("signature", None)
        encoded = json.dumps(
            {
                "domain": "euroeval-volunteer-marker",
                "version": 1,
                "issue_number": issue_number,
                "marker": unsigned,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        expected = (
            base64.urlsafe_b64encode(
                hmac.new(secret.encode(), encoded, hashlib.sha256).digest()
            )
            .decode()
            .rstrip("=")
        )
        if not hmac.compare_digest(signature, expected):
            return None
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
    if len({item["submission_id"] for item in submissions}) != len(submissions):
        return None
    return CommunityMarker(
        1,
        payload["coordinator"],
        payload["submission"],
        tuple(leases),
        tuple(submissions),
        tuple(completed),
        signature,
    )


def _expiry_active(value: str) -> bool:
    try:
        return dt.datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ) > dt.datetime.now(dt.UTC)
    except ValueError:
        return False


def issue_has_active_queue_ownership(body: str) -> bool:
    """Return whether a valid, unexpired coordinator marker protects an issue."""
    marker = parse_community_marker(body)
    return (
        marker is not None
        and marker.submission in COMMUNITY_ACTIVE_SUBMISSION_STATES
        and (
            marker.submission == "submitted"
            or marker.submission == "active"
            and (
                not marker.leases
                or any(_expiry_active(lease["expires_at"]) for lease in marker.leases)
            )
        )
    )


def issue_has_community_marker(body: str) -> bool:
    """Return whether an issue contains any broker marker candidate."""
    return bool(_COMMUNITY_MARKER_CANDIDATE_RE.search(body))


def issue_has_terminal_queue_submission(
    body: str, *, issue_number: int | None = None
) -> bool:
    """Return whether a marker records a trusted terminal submission.

    Rejected markers remain auditable even when their signature cannot be
    verified. Accepted markers can protect local queue ownership only when
    their issue-bound signature is verified.
    """
    marker = parse_community_marker(body)
    if marker is None:
        return False
    if marker.submission == "rejected":
        return True
    return (
        marker.submission == "accepted"
        and issue_number is not None
        and _trusted_issue_marker(issue_number, body) is not None
    )


def release_issue_if_owned(number: int, vm_id: str, assignee: str) -> bool:
    """Release only after both marker clearing and unassignment succeed.

    Returns:
        Whether both GitHub mutations succeeded.
    """
    body = fetch_issue_body(number=number)
    if (
        _COMMUNITY_MARKER_CANDIDATE_RE.search(body)
        and _trusted_issue_marker(number, body) is None
    ):
        return False
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
    marker = _trusted_issue_marker(number=number, body=body)
    if _COMMUNITY_MARKER_CANDIDATE_RE.search(body) and marker is None:
        return False
    if marker is not None and marker.submission in COMMUNITY_ACTIVE_SUBMISSION_STATES:
        active = any(_expiry_active(lease["expires_at"]) for lease in marker.leases)
        if active or not marker.leases or marker.signature:
            return False
        if not marker.submissions:
            body = remove_community_marker(body)
    match = VM_MARKER_RE.search(body)
    if match and match.group(1) != vm_id:
        return False
    # Keep signed broker state and all submission history as an audit trail.
    # Only an unsigned, expired legacy lease with no history is discarded.
    cleaned = VM_MARKER_RE.sub("", body).rstrip()
    patch_issue_body(number=number, body=f"{cleaned}\n\n<!-- vm-id: {vm_id} -->\n")
    return True


def remove_community_marker(body: str) -> str:
    """Return ``body`` without its recognised canonical marker."""
    if parse_community_marker(body) is None:
        return body
    return COMMUNITY_MARKER_RE.sub("", body, count=1).rstrip() + "\n"


def vm_marker_matches(number: int, vm_id: str) -> bool:
    """Return whether the local VM marker is still safe to touch."""
    body = fetch_issue_body(number=number)
    if issue_has_active_queue_ownership(body):
        return False
    match = VM_MARKER_RE.search(body)
    return match is None or match.group(1) == vm_id
