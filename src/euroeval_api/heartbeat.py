"""Worker heartbeat endpoint for renewing owned evaluation leases."""

from __future__ import annotations

import datetime as dt
import typing as t

from fastapi import APIRouter, HTTPException, Request

from .auth import authenticate, enforce_rate_limit
from .dependencies import (
    PROTOCOL_VERSION,
    get_http,
    get_redis,
    read_json,
    require_protocol,
)
from .github import GitHub, require_lease_ownership
from .leases import require_live_lease
from .markers import (
    VolunteerMarker,
    parse_volunteer_marker,
    replace_volunteer_marker,
    sign_volunteer_marker,
    verify_volunteer_marker,
)
from .state import acquire_renewable_issue_mutex, lease_ttl, save_lease

router = APIRouter()


@router.post("/api/worker/heartbeat")
async def heartbeat(request: Request) -> dict[str, object]:
    """Renew a worker lease only while its GitHub and Redis fences remain current.

    Args:
        request: Incoming worker heartbeat request.

    Returns:
        Protocol metadata and the renewed lease expiry.

    Raises:
        HTTPException: If authentication, lease ownership, or renewal fails.
    """
    identity = await authenticate(request=request)
    redis = await get_redis(request=request)
    await enforce_rate_limit(
        redis=redis,
        key=f"euroeval:worker:limit:heartbeat:{identity['hash']}",
        limit=240,
        window_seconds=3600,
    )
    body = await read_json(request=request, limit=8 * 1024)
    require_protocol(body=body)
    lease_id = body.get("lease_id")
    if not isinstance(lease_id, str):
        raise HTTPException(status_code=400, detail="lease_id is required.")
    if len(lease_id) > 400:
        raise HTTPException(status_code=400, detail="lease_id is too long.")
    lease = await require_live_lease(
        redis=redis, lease_id=lease_id, contributor=identity["contributor"]
    )
    mutex = await acquire_renewable_issue_mutex(
        redis=redis, issue=lease["issue_number"]
    )
    if mutex is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "Issue is busy; retry heartbeat.", "code": "issue_busy"},
        )
    try:
        github = GitHub(http=get_http(request=request))
        issue = await github.get_issue(number=lease["issue_number"])
        require_lease_ownership(issue=issue, lease=lease)
        marker = parse_volunteer_marker(body=issue.get("body"))
        marker_lease = (
            next(
                (item for item in marker["leases"] if item["lease_id"] == lease_id),
                None,
            )
            if marker
            else None
        )
        if (
            marker is None
            or marker_lease is None
            or not verify_volunteer_marker(issue_number=issue["number"], marker=marker)
        ):
            raise HTTPException(
                status_code=409,
                detail="GitHub ownership marker is missing, unsigned, or malformed.",
            )
        expires_at = (
            (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=lease_ttl()))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        updated = t.cast(
            VolunteerMarker,
            {
                **marker,
                "leases": [
                    {**item, "expires_at": expires_at}
                    if item["lease_id"] == lease_id
                    else item
                    for item in marker["leases"]
                ],
            },
        )
        signed = sign_volunteer_marker(
            issue_number=lease["issue_number"], marker=updated
        )
        original_expiry = lease["expires_at"]
        if dt.datetime.fromisoformat(
            original_expiry.replace("Z", "+00:00")
        ) <= dt.datetime.now(dt.timezone.utc):
            raise HTTPException(
                status_code=409,
                detail="Lease is absent, expired, or belongs to another contributor.",
            )
        await mutex.assert_owned()
        await github.patch_issue(
            number=lease["issue_number"],
            body=replace_volunteer_marker(body=issue.get("body") or "", marker=signed),
        )
        fenced_issue = await github.get_issue(number=lease["issue_number"])
        fenced = parse_volunteer_marker(body=fenced_issue.get("body"))
        if (
            fenced is None
            or not verify_volunteer_marker(
                issue_number=lease["issue_number"], marker=fenced
            )
            or not any(
                item["lease_id"] == lease_id and item["expires_at"] == expires_at
                for item in fenced["leases"]
            )
        ):
            raise HTTPException(status_code=409, detail="GitHub heartbeat fence lost.")
        require_lease_ownership(issue=fenced_issue, lease=lease)
        if lease.get("ownership_mode") == "public":
            await github.reconcile_public_assignments(
                redis=redis,
                issue=fenced_issue,
                marker=fenced,
                assert_owned=mutex.assert_owned,
            )
        lease["expires_at"] = expires_at
        if not await save_lease(redis=redis, lease=lease):
            raise HTTPException(
                status_code=409,
                detail="Lease was replaced or expired before it could be renewed.",
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "lease_id": lease_id,
            "expires_at": expires_at,
        }
    finally:
        await mutex.release()
