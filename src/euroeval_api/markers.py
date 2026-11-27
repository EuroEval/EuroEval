"""Signed issue markers and their public/legacy ownership rules."""

from __future__ import annotations

import base64
import hashlib
import hmac
import math
import re
import time
import typing as t
from datetime import UTC, datetime

import rfc8785
from pydantic_core import from_json

from .dependencies import PROTOCOL_VERSION, require_env

TITLE_PREFIX = "[MODEL EVALUATION REQUEST]"
VOLUNTEER_MARKER_RE = re.compile(
    r"<!--[ \t]*euroeval-volunteer-worker:v1[ \t]+([^<]*?)-->", re.I
)
GROUPS = {
    "Baltic languages (Latvian, Lithuanian)": ["lv", "lt"],
    "Finnic languages (Estonian, Finnish)": ["et", "fi"],
    "Romance languages (Catalan, French, Italian, Portuguese, Romanian, Spanish)": [
        "ca",
        "fr",
        "it",
        "pt",
        "ro",
        "es",
    ],
    "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)": [
        "da",
        "fo",
        "is",
        "no",
        "sv",
    ],
    (
        "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech, Polish, "
        "Serbian, Slovak, Slovenian, Ukrainian)"
    ): ["be", "bg", "bs", "hr", "cs", "pl", "sr", "sk", "sl", "uk"],
    "West Germanic languages (Dutch, English, German, Luxembourgish)": [
        "nl",
        "en",
        "de",
        "lb",
    ],
    "Albanian": ["sq"],
    "Greek": ["el"],
    "Hungarian": ["hu"],
}


class MarkerLease(t.TypedDict):
    """The signed projection of a persisted worker lease."""

    lease_id: str
    language: str
    worker: str
    contributor: str
    expires_at: str
    ownership_mode: t.NotRequired[t.Literal["assignee", "public"]]


class Submission(t.TypedDict):
    """A staged submission's immutable identity and review state."""

    submission_id: str
    language: str
    manifest_path: str
    submitted_at: str
    verified_contributor: str
    result_count: int
    status: t.Literal["submitted", "accepted", "rejected"]
    ownership_mode: t.NotRequired[t.Literal["assignee", "public"]]


class VolunteerMarker(t.TypedDict):
    """The marker wire format, including optional historical fields."""

    protocol_version: str
    coordinator: str
    submission: t.Literal["active", "submitted", "accepted", "rejected"]
    leases: list[MarkerLease]
    submissions: t.NotRequired[list[Submission]]
    completed_languages: t.NotRequired[list[str]]
    public_assignees: t.NotRequired[list[str]]
    signature: t.NotRequired[str]


type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)


def claimable_languages(
    selected: list[str], marker: VolunteerMarker | None, now: float | None = None
) -> list[str]:
    """Exclude live leases and already submitted or accepted language results.

    Args:
        selected: Requested languages.
        marker: Current ownership evidence.
        now (optional): Current Unix milliseconds.

    Returns:
        Remaining claimable languages.
    """
    if marker is None:
        return selected
    current = time.time() * 1000 if now is None else now
    return [
        language
        for language in selected
        if not any(
            value["language"] == language
            and timestamp(value=value["expires_at"]) > current
            for value in marker["leases"]
        )
        and not any(
            value["language"] == language
            and value["status"] in ("submitted", "accepted")
            for value in marker.get("submissions", [])
        )
    ]


def timestamp(value: str) -> float:
    """Read a persisted ISO date with Date.parse-style invalid-date semantics.

    Args:
        value: Persisted date string.

    Returns:
        Unix milliseconds, or NaN for an invalid date.
    """
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (date if date.tzinfo else date.replace(tzinfo=UTC)).timestamp() * 1000
    except (ValueError, OverflowError):
        return float("nan")


def extract_model_id(title: str, body: str | None) -> str | None:
    """Read a syntactically valid model ID from the issue's form or title.

    Args:
        title: Issue title.
        body: Issue body.

    Returns:
        Model ID or None when the issue does not identify one.
    """
    match = re.search(r"(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)", body or "", re.I)
    if not match and not title.startswith(TITLE_PREFIX + " "):
        return None
    value = (
        (match.group(1) if match else title[len(TITLE_PREFIX) :]).strip().strip("`*_")
    )
    return (
        value
        if re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?::[A-Z0-9_]+)?", value)
        else None
    )


def has_volunteer_marker(body: str | None) -> bool:
    """Detect marker syntax even when its payload cannot be validated.

    Args:
        body: GitHub issue body.

    Returns:
        Whether marker syntax is present.
    """
    return bool(VOLUNTEER_MARKER_RE.search(body or ""))


def is_ready_for_review(marker: VolunteerMarker, selected: list[str]) -> bool:
    """Require complete requested coverage and at least one pending review.

    Args:
        marker: Current ownership/submission evidence.
        selected: Requested languages.

    Returns:
        Whether the request is ready for review.
    """
    submissions = marker.get("submissions", [])
    completed = {
        value["language"]
        for value in submissions
        if value["status"] in ("submitted", "accepted")
    }
    return (
        bool(selected)
        and all(value in completed for value in selected)
        and any(
            value["status"] == "submitted" and value["language"] in selected
            for value in submissions
        )
    )


def parse_volunteer_marker(body: str | None) -> VolunteerMarker | None:
    """Read exactly one valid ownership marker, preserving signed nested extensions.

    Args:
        body: GitHub issue body.

    Returns:
        Validated marker, or None for absent, duplicate, or malformed evidence.
    """
    matches = list(VOLUNTEER_MARKER_RE.finditer(body or ""))
    if len(matches) != 1:
        return None
    try:
        item = from_json(matches[0].group(1).strip(), allow_inf_nan=False)
        if (
            not isinstance(item, dict)
            or item.get("protocol_version") != PROTOCOL_VERSION
            or not isinstance(item.get("coordinator"), str)
            or not item["coordinator"]
            or item.get("submission")
            not in ("active", "submitted", "accepted", "rejected")
            or not isinstance(item.get("leases"), list)
        ):
            return None
        leases = item["leases"]
        if (
            not all(_valid_lease(value=lease) for lease in leases)
            or len({lease["lease_id"] for lease in leases}) != len(leases)
            or len({lease["language"] for lease in leases}) != len(leases)
        ):
            return None
        if "submissions" in item:
            submissions = item["submissions"]
            if (
                not isinstance(submissions, list)
                or not all(_valid_submission(value=value) for value in submissions)
                or len({value["submission_id"] for value in submissions})
                != len(submissions)
            ):
                return None
        for key in ("completed_languages", "public_assignees"):
            if key in item and (
                not isinstance(item[key], list)
                or any(
                    not isinstance(value, str)
                    or not value
                    or (key == "public_assignees" and not value.strip())
                    for value in item[key]
                )
            ):
                return None
        if "public_assignees" in item and (
            len({value.lower() for value in item["public_assignees"]})
            != len(item["public_assignees"])
        ):
            return None
        result = {
            key: item[key]
            for key in (
                "protocol_version",
                "coordinator",
                "submission",
                "leases",
                "submissions",
                "completed_languages",
                "public_assignees",
            )
            if key in item
        }
        if isinstance(item.get("signature"), str):
            result["signature"] = item["signature"]
        return t.cast(VolunteerMarker, result)
    except (ValueError, TypeError, KeyError, OverflowError):
        return None


def _valid_lease(value: object) -> bool:
    """Validate the required signed lease fields without discarding extensions.

    Args:
        value: Decoded lease entry.

    Returns:
        Whether the entry satisfies the deployed marker format.
    """
    return (
        isinstance(value, dict)
        and all(
            isinstance(value.get(key), str)
            for key in ("lease_id", "language", "worker", "contributor", "expires_at")
        )
        and (
            "ownership_mode" not in value
            or value["ownership_mode"] in ("assignee", "public")
        )
        and math.isfinite(timestamp(value["expires_at"]))
    )


def _valid_submission(value: object) -> bool:
    """Validate staged identity, count, date, and ownership evidence.

    Args:
        value: Decoded submission entry.

    Returns:
        Whether every required submission field is valid.
    """
    if not isinstance(value, dict):
        return False
    count = value.get("result_count")
    return (
        all(
            isinstance(value.get(key), str) and value[key]
            for key in (
                "submission_id",
                "language",
                "manifest_path",
                "submitted_at",
                "verified_contributor",
            )
        )
        and math.isfinite(timestamp(value["submitted_at"]))
        and isinstance(count, (int, float))
        and not isinstance(count, bool)
        and 0 < count <= 9007199254740991
        and int(count) == count
        and (
            "ownership_mode" not in value
            or value["ownership_mode"] in ("assignee", "public")
        )
        and value.get("status") in ("submitted", "accepted", "rejected")
    )


def replace_volunteer_marker(body: str, marker: VolunteerMarker | None) -> str:
    """Replace ownership evidence while retaining the surrounding request text.

    Args:
        body: Original issue body.
        marker: Replacement marker, or None to remove it.

    Returns:
        Updated issue body.
    """
    without = VOLUNTEER_MARKER_RE.sub("", remove_final_credit(body=body), count=1)
    without = re.sub(r"\n{3,}", "\n\n", without).rstrip()
    return (
        f"{without}\n\n{render_volunteer_marker(marker=marker)}\n"
        if marker
        else f"{without}\n"
    )


def remove_final_credit(body: str) -> str:
    """Remove obsolete final-credit comments when replacing ownership evidence.

    Args:
        body: GitHub issue body.

    Returns:
        Remaining body with collapsed blank lines.
    """
    without = re.sub(
        r"<!--[ \t]*euroeval-volunteer-credit:v1[\s\S]*?-->", "", body, flags=re.I
    )
    return re.sub(r"\n{3,}", "\n\n", without).rstrip()


def render_volunteer_marker(marker: VolunteerMarker) -> str:
    """Render the deployed marker comment format.

    Args:
        marker: Signed or unsigned ownership marker.

    Returns:
        Canonical marker comment.
    """
    return f"<!-- euroeval-volunteer-worker:v1 {canonical_json(value=marker)} -->"


def canonical_json(value: object) -> str:
    """Encode hash/signature evidence using JavaScript numeric and key ordering.

    Args:
        value: JSON-compatible protocol evidence.

    Returns:
        Canonical UTF-8 JSON text.
    """
    return rfc8785.dumps(t.cast(JsonValue, value)).decode("utf-8")


def selected_languages(body: str | None) -> list[str]:
    """Expand checked language groups without altering their requested order.

    Args:
        body: Request issue body.

    Returns:
        Selected language codes.
    """
    return [
        code
        for group, codes in GROUPS.items()
        if re.search(
            r"-\s*\[[xX]\]\s*" + re.escape(group) + r"(?:\s|$)", body or "", re.M
        )
        for code in codes
    ]


def verify_volunteer_marker(issue_number: int, marker: VolunteerMarker) -> bool:
    """Verify persisted evidence and its issue binding without timing leaks.

    Args:
        issue_number: Issue whose ownership is being checked.
        marker: Parsed signed evidence.

    Returns:
        Whether the signature matches the deployed format and configured secret.
    """
    signature = marker.get("signature")
    return bool(signature) and hmac.compare_digest(
        sign_volunteer_marker(issue_number=issue_number, marker=marker)["signature"],
        signature,
    )


def sign_volunteer_marker(
    issue_number: int, marker: VolunteerMarker
) -> VolunteerMarker:
    """Sign canonical marker evidence bound to its GitHub issue.

    Args:
        issue_number: Issue number included in the signature domain.
        marker: Ownership evidence, with any old signature ignored.

    Returns:
        Marker with its URL-safe, unpadded HMAC signature.
    """
    payload = {
        "domain": "euroeval-volunteer-marker",
        "version": 1,
        "issue_number": issue_number,
        "marker": {key: value for key, value in marker.items() if key != "signature"},
    }
    digest = hmac.new(
        require_env(name="VOLUNTEER_MARKER_SECRET").encode(),
        canonical_json(value=payload).encode(),
        hashlib.sha256,
    ).digest()
    return {
        **marker,
        "signature": base64.urlsafe_b64encode(digest).decode().rstrip("="),
    }


def volunteer_assignees_match(
    assignees: list[dict[str, str]] | None,
    marker: VolunteerMarker | None,
    now: float | None = None,
    include_expired_leases: bool = False,
    allow_expired_unassigned: bool = False,
) -> bool:
    """Check legacy assignments without treating public projections as ownership.

    Args:
        assignees: Current GitHub assignees.
        marker: Parsed ownership evidence.
        now (optional): Current Unix milliseconds.
        include_expired_leases (optional): Include ownership awaiting recovery.
        allow_expired_unassigned (optional): Allow already-unassigned expired owners.

    Returns:
        Whether the assignments are consistent with signed ownership.
    """
    current = time.time() * 1000 if now is None else now
    actual = {value["login"].lower() for value in assignees or []}
    if marker is None:
        return not actual
    expected: set[str] = set()
    optional: set[str] = set()
    submissions = marker.get("submissions", [])
    for lease in marker["leases"]:
        if lease.get("ownership_mode") == "public":
            continue
        who = lease["contributor"].lower()
        if timestamp(value=lease["expires_at"]) > current:
            expected.add(who)
        elif include_expired_leases:
            expected.add(who)
            retained = any(
                other["lease_id"] != lease["lease_id"]
                and other.get("ownership_mode") != "public"
                and timestamp(value=other["expires_at"]) > current
                and other["contributor"].lower() == who
                for other in marker["leases"]
            ) or any(
                value["status"] in ("submitted", "accepted")
                and value.get("ownership_mode") != "public"
                and value["verified_contributor"].lower() == who
                for value in submissions
            )
            if allow_expired_unassigned and not retained:
                optional.add(who)
    expected.update(
        value["verified_contributor"].lower()
        for value in submissions
        if value["status"] in ("submitted", "accepted")
        and value.get("ownership_mode") != "public"
    )
    expected.update(value.lower() for value in marker.get("public_assignees", []))
    public = any(
        value.get("ownership_mode") == "public" for value in marker["leases"]
    ) or any(value.get("ownership_mode") == "public" for value in submissions)
    return (public or actual <= expected) and expected <= actual | optional
