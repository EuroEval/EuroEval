"""Release a worker lease while preserving signed GitHub ownership."""

from __future__ import annotations

import math
import time
import typing as t

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from . import markers, state
from .auth import authenticate, enforce_rate_limit
from .dependencies import (
    PROTOCOL_VERSION,
    get_http,
    get_redis,
    read_json,
    require_protocol,
)
from .github import GitHub, GithubIssue
from .leases import Lease

router = APIRouter()


@router.post("/api/worker/release")
async def release_lease(request: Request) -> JSONResponse:
    """Release the authenticated contributor's lease and its GitHub projection.

    Args:
        request: Incoming worker request.

    Returns:
        Protocol response confirming release.

    Raises:
        HTTPException: Invalid, stale, busy, or unowned lease.
        _failure: Builds protocol-specific HTTP errors.
    """
    identity = await authenticate(request=request)
    redis = await get_redis(request=request)
    await enforce_rate_limit(
        redis=redis,
        key=f"euroeval:worker:limit:release:{identity['hash']}",
        limit=60,
        window_seconds=3600,
    )
    body = await read_json(request=request, limit=8192)
    require_protocol(body=body)
    lease_id = body.get("lease_id")
    if not isinstance(lease_id, str):
        raise HTTPException(status_code=400, detail="lease_id is required.")
    stored = await state.get_lease_by_id(redis=redis, lease_id=lease_id)
    lease = t.cast(Lease | None, stored)
    if (
        lease
        and lease.get("released") is True
        and lease.get("contributor", "").casefold()
        == identity["contributor"].casefold()
    ):
        return JSONResponse(
            {
                "protocol_version": PROTOCOL_VERSION,
                "status": "released",
                "lease_id": lease_id,
            }
        )
    if (
        lease is None
        or lease.get("released") is True
        or lease.get("contributor", "").casefold() != identity["contributor"].casefold()
        or not math.isfinite(markers.timestamp(lease["expires_at"]))
        or markers.timestamp(lease["expires_at"]) <= time.time() * 1000
    ):
        raise _failure(
            409, "Lease is absent, expired, or belongs to another contributor."
        )
    mutex = await state.acquire_renewable_issue_mutex(
        redis=redis, issue=lease["issue_number"]
    )
    if mutex is None:
        raise _failure(503, "Issue is busy; retry release.", "ISSUE_BUSY")
    try:
        github = GitHub(http=get_http(request=request))
        issue = await github.get_issue(number=lease["issue_number"])
        marker = markers.parse_volunteer_marker(body=issue.get("body"))
        if marker is None:
            if (
                markers.has_volunteer_marker(issue.get("body"))
                or lease.get("ownership_mode") == "public"
            ):
                raise _failure(
                    409, "GitHub ownership marker is missing, unsigned, or malformed."
                )
            await state.delete_lease(redis=redis, lease=lease)
            return JSONResponse(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "status": "released",
                    "lease_id": lease_id,
                }
            )
        if not markers.verify_volunteer_marker(
            issue_number=lease["issue_number"], marker=marker
        ):
            raise _failure(
                409, "GitHub ownership marker is missing, unsigned, or malformed."
            )
        entry = next(
            (item for item in marker["leases"] if item["lease_id"] == lease_id), None
        )
        if entry is None:
            await state.delete_lease(redis=redis, lease=lease)
        else:
            if entry["contributor"].casefold() != lease[
                "contributor"
            ].casefold() or entry.get("ownership_mode", "assignee") != lease.get(
                "ownership_mode", "assignee"
            ):
                raise _failure(409, "GitHub release ownership fence lost.")
            retained = any(
                item["lease_id"] != lease_id
                and item.get("ownership_mode", "assignee") != "public"
                and item["contributor"].casefold() == identity["contributor"].casefold()
                for item in marker["leases"]
            ) or any(
                item["status"] in ("submitted", "accepted")
                and item.get("ownership_mode", "assignee") != "public"
                and item["verified_contributor"].casefold()
                == identity["contributor"].casefold()
                for item in marker.get("submissions", [])
            )
            if (
                not _matches(
                    issue, _expected(marker), bool(marker.get("public_assignees"))
                )
                and not retained
            ):
                raise _failure(409, "GitHub release assignment changed; retry release.")
            if (
                lease.get("ownership_mode") != "public"
                and not retained
                and any(
                    item["login"].casefold() == identity["contributor"].casefold()
                    for item in issue.get("assignees", []) or []
                )
            ):
                await mutex.assert_owned()
                try:
                    await github.unassign_issue(
                        number=lease["issue_number"], login=identity["contributor"]
                    )
                except Exception:
                    observed = await github.get_issue(number=lease["issue_number"])
                    if any(
                        item["login"].casefold() == identity["contributor"].casefold()
                        for item in observed.get("assignees", []) or []
                    ):
                        raise
                issue = await github.get_issue(number=lease["issue_number"])
            issue = await github.get_issue(number=lease["issue_number"])
            marker = markers.parse_volunteer_marker(body=issue.get("body"))
            if (
                marker is None
                or not markers.verify_volunteer_marker(
                    issue_number=lease["issue_number"], marker=marker
                )
                or not any(item["lease_id"] == lease_id for item in marker["leases"])
            ):
                raise _failure(409, "GitHub release fence lost.")
            remaining = [
                item for item in marker["leases"] if item["lease_id"] != lease_id
            ]
            next_marker: markers.VolunteerMarker = {**marker, "leases": remaining}
            next_marker["submission"] = (
                marker["submission"]
                if marker["submission"] in ("accepted", "rejected")
                else "submitted"
                if marker.get("submissions")
                else "active"
            )
            signed = (
                markers.sign_volunteer_marker(
                    issue_number=lease["issue_number"], marker=next_marker
                )
                if remaining
                or marker.get("submissions")
                or marker.get("completed_languages")
                or marker.get("public_assignees")
                or marker["submission"] in ("accepted", "rejected")
                else None
            )
            await mutex.assert_owned()
            await github.patch_issue(
                number=lease["issue_number"],
                body=markers.replace_volunteer_marker(
                    body=issue.get("body") or "", marker=signed
                ),
            )
            await state.delete_lease(redis=redis, lease=lease)
    finally:
        await mutex.release()
    return JSONResponse(
        {
            "protocol_version": PROTOCOL_VERSION,
            "status": "released",
            "lease_id": lease_id,
        }
    )


def _expected(marker: markers.VolunteerMarker) -> set[str]:
    """Find legacy assignments retained by signed evidence.

    Args:
        marker: Verified ownership marker.

    Returns:
        Contributor logins still entitled to legacy assignment.
    """
    values = {
        lease["contributor"].casefold()
        for lease in marker["leases"]
        if lease.get("ownership_mode", "assignee") != "public"
    }
    values.update(
        item["verified_contributor"].casefold()
        for item in marker.get("submissions", [])
        if item["status"] in ("submitted", "accepted")
        and item.get("ownership_mode", "assignee") != "public"
    )
    return values


def _failure(status: int, message: str, code: str | None = None) -> HTTPException:
    """Build a protocol-shaped broker error.

    Args:
        status: HTTP response status.
        message: Public error text.
        code: Optional stable error code.

    Returns:
        A FastAPI exception carrying the public response fields.
    """
    detail: dict[str, str] = {"error": message, "protocol_version": PROTOCOL_VERSION}
    if code:
        detail["code"] = code
    return HTTPException(status_code=status, detail=detail)


def _matches(issue: GithubIssue, expected: set[str], allow_extra: bool) -> bool:
    """Compare GitHub assignments against signed ownership.

    Args:
        issue: Current GitHub issue.
        expected: Expected contributor logins.
        allow_extra: Whether public projections may add assignees.

    Returns:
        Whether the observed assignment set is safe.
    """
    actual = {item["login"].casefold() for item in issue.get("assignees", []) or []}
    return expected <= actual and (allow_extra or expected == actual)
