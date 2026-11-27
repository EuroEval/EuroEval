"""Release leases only after observing their signed GitHub ownership transition."""

from __future__ import annotations

import math
import time
import typing as t

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from upstash_redis.asyncio import Redis

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
    """Release a contributor's lease without discarding unrelated signed history.

    Args:
        request:
            Authenticated worker release request.

    Returns:
        Idempotent release receipt.

    Raises:
        HTTPException:
            If ownership, assignment or cleanup cannot be established.
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
    lease = t.cast(
        Lease | None, await state.get_lease_by_id(redis=redis, lease_id=lease_id)
    )
    receipt = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "released",
        "lease_id": lease_id,
    }
    owned = (
        lease is not None
        and lease["contributor"].lower() == identity["contributor"].lower()
    )
    if owned and lease and lease.get("released") is True:
        return JSONResponse(receipt)
    if (
        not owned
        or not lease
        or lease.get("released")
        or not math.isfinite(markers.timestamp(value=lease["expires_at"]))
        or markers.timestamp(value=lease["expires_at"]) <= time.time() * 1000
    ):
        raise HTTPException(
            status_code=409,
            detail="Lease is absent, expired, or belongs to another contributor.",
        )
    mutex = await state.acquire_renewable_issue_mutex(
        redis=redis, issue=lease["issue_number"]
    )
    if mutex is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "Issue is busy; retry release.", "code": "ISSUE_BUSY"},
        )
    try:
        active = await state.get_lease_for_issue(
            redis=redis, issue=lease["issue_number"], language=lease["language"]
        )
        if active and active["lease_id"] != lease_id:
            raise HTTPException(
                status_code=409, detail="A different lease owns this language."
            )
        github = GitHub(http=get_http(request=request))
        issue = await github.get_issue(number=lease["issue_number"])
        marker = _verified(issue=issue)
        if marker is None:
            if lease.get("ownership_mode") == "public":
                raise HTTPException(
                    status_code=409, detail="Public lease ownership marker is missing."
                )
        elif any(item["lease_id"] == lease_id for item in marker["leases"]):
            await _transition(
                github=github,
                redis=redis,
                mutex=mutex,
                lease=lease,
                issue=issue,
                marker=marker,
            )
        elif lease.get("ownership_mode") == "public":
            await _reconcile(
                github=github,
                redis=redis,
                mutex=mutex,
                lease=lease,
                issue=issue,
                marker=marker,
            )
        await state.delete_lease(redis=redis, lease=lease)
    finally:
        await mutex.release()
    return JSONResponse(receipt)


async def _reconcile(
    github: GitHub,
    redis: Redis,
    mutex: state.IssueMutex,
    lease: Lease,
    issue: GithubIssue,
    marker: markers.VolunteerMarker,
) -> None:
    """Finish public projection cleanup before deleting its lease reservation.

    Args:
        github:
            Issue provider.
        redis:
            Authoritative lease store.
        mutex:
            Held issue mutex.
        lease:
            Lease being released.
        issue:
            Current issue snapshot.
        marker:
            Verified signed evidence.

    Raises:
        HTTPException:
            If this contributor's public removal remains pending.
    """
    result = await github.reconcile_public_assignments(
        redis=redis, issue=issue, marker=marker, assert_owned=mutex.assert_owned
    )
    if lease.get("ownership_mode") == "public" and any(
        login.lower() == lease["contributor"].lower()
        for login in result["pending_removals"]
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Public assignment cleanup is pending; retry release.",
                "code": "release_cleanup_pending",
            },
        )


async def _transition(
    github: GitHub,
    redis: Redis,
    mutex: state.IssueMutex,
    lease: Lease,
    issue: GithubIssue,
    marker: markers.VolunteerMarker,
) -> None:
    """Remove only this slot, retaining all other assignment entitlements.

    Args:
        github:
            Issue provider.
        redis:
            Authoritative lease store.
        mutex:
            Held issue mutex.
        lease:
            Contributor-owned lease.
        issue:
            Current issue snapshot.
        marker:
            Verified signed evidence.

    Raises:
        HTTPException:
            If legacy assignment or signed ownership changes.
    """
    _require_slot(lease=lease, marker=marker)
    contributor = lease["contributor"].lower()
    expected = _expected(marker=marker)
    retained = contributor in _expected(
        marker={
            **marker,
            "leases": [
                item
                for item in marker["leases"]
                if item["lease_id"] != lease["lease_id"]
            ],
        }
    )
    without = expected if retained else expected - {contributor}
    allow_extra = _has_public(marker=marker)
    if not _matches(issue=issue, expected=expected, allow_extra=allow_extra) and (
        retained or not _matches(issue=issue, expected=without, allow_extra=allow_extra)
    ):
        raise HTTPException(
            status_code=409, detail="GitHub release assignment changed; retry release."
        )
    assigned = next(
        (
            item["login"]
            for item in issue.get("assignees", []) or []
            if item["login"].lower() == contributor
        ),
        None,
    )
    if lease.get("ownership_mode") != "public" and assigned and not retained:
        await mutex.assert_owned()
        try:
            await github.unassign_issue(number=lease["issue_number"], login=assigned)
        except Exception:
            observed = await _observe(github=github, number=lease["issue_number"])
            if observed is None or any(
                item["login"].lower() == contributor
                for item in observed.get("assignees", []) or []
            ):
                raise
        issue = await github.get_issue(number=lease["issue_number"])
        checked = _verified(issue=issue)
        if checked is None or any(
            item["login"].lower() == contributor
            for item in issue.get("assignees", []) or []
        ):
            raise HTTPException(status_code=409, detail="GitHub release fence lost.")
        _require_slot(lease=lease, marker=checked)
    issue = await github.get_issue(number=lease["issue_number"])
    checked = _verified(issue=issue)
    if checked is None:
        raise HTTPException(status_code=409, detail="GitHub release fence lost.")
    _require_slot(lease=lease, marker=checked)
    remaining = [
        item for item in checked["leases"] if item["lease_id"] != lease["lease_id"]
    ]
    if not _matches(
        issue=issue,
        expected=_expected(marker={**checked, "leases": remaining}),
        allow_extra=_has_public(marker=checked),
    ):
        raise HTTPException(
            status_code=409, detail="GitHub release assignment changed; retry release."
        )
    terminal = checked["submission"] in {"accepted", "rejected"}
    next_marker: markers.VolunteerMarker | None = None
    if (
        remaining
        or checked.get("submissions")
        or checked.get("completed_languages")
        or checked.get("public_assignees")
        or terminal
    ):
        next_marker = {**checked, "leases": remaining}
        next_marker["submission"] = (
            checked["submission"]
            if terminal
            else "submitted"
            if checked.get("submissions")
            else "active"
        )
        next_marker = markers.sign_volunteer_marker(
            issue_number=lease["issue_number"], marker=next_marker
        )
    await mutex.assert_owned()
    try:
        await github.patch_issue(
            number=lease["issue_number"],
            body=markers.replace_volunteer_marker(
                body=issue["body"] or "", marker=next_marker
            ),
        )
    except Exception:
        observed = await _observe(github=github, number=lease["issue_number"])
        if observed is None:
            raise
        observed_marker = _verified(issue=observed)
        if observed_marker and any(
            item["lease_id"] == lease["lease_id"] for item in observed_marker["leases"]
        ):
            raise
    issue = await github.get_issue(number=lease["issue_number"])
    final = _verified(issue=issue)
    if final and any(item["lease_id"] == lease["lease_id"] for item in final["leases"]):
        raise HTTPException(status_code=409, detail="GitHub release fence lost.")
    if not _matches(
        issue=issue,
        expected=_expected(marker=final) if final else set(),
        allow_extra=_has_public(marker=final) if final else False,
    ):
        raise HTTPException(
            status_code=409, detail="GitHub release assignment changed; retry release."
        )
    if final:
        await _reconcile(
            github=github,
            redis=redis,
            mutex=mutex,
            lease=lease,
            issue=issue,
            marker=final,
        )


def _expected(marker: markers.VolunteerMarker) -> set[str]:
    """Project legacy assignments required by leases and retained submissions.

    Args:
        marker:
            Verified signed evidence.

    Returns:
        Required lower-case GitHub logins.
    """
    return {
        item["contributor"].lower()
        for item in marker["leases"]
        if item.get("ownership_mode") != "public"
    } | {
        item["verified_contributor"].lower()
        for item in marker.get("submissions", [])
        if item["status"] in {"submitted", "accepted"}
        and item.get("ownership_mode") != "public"
    }


def _has_public(marker: markers.VolunteerMarker) -> bool:
    """Allow unrelated assignments when signed public ownership is present.

    Args:
        marker:
            Verified signed evidence.

    Returns:
        Whether the marker permits additional assignees.
    """
    return (
        bool(marker.get("public_assignees"))
        or any(item.get("ownership_mode") == "public" for item in marker["leases"])
        or any(
            item.get("ownership_mode") == "public"
            for item in marker.get("submissions", [])
        )
    )


def _matches(issue: GithubIssue, expected: set[str], allow_extra: bool) -> bool:
    """Compare the observed assignment set against required legacy owners.

    Args:
        issue:
            Observed issue.
        expected:
            Required logins.
        allow_extra:
            Whether public ownership permits other assignments.

    Returns:
        Whether all required assignments remain safe.
    """
    actual = {item["login"].lower() for item in issue.get("assignees", []) or []}
    return expected <= actual and (allow_extra or expected == actual)


async def _observe(github: GitHub, number: int) -> GithubIssue | None:
    """Observe an ambiguous mutation without treating an outage as success.

    Args:
        github:
            Issue provider.
        number:
            Issue identity.

    Returns:
        Observed issue, or none if observation failed.
    """
    try:
        return await github.get_issue(number=number)
    except Exception:
        return None


def _require_slot(lease: Lease, marker: markers.VolunteerMarker) -> None:
    """Require the exact slot's contributor and ownership mode before removal.

    Args:
        lease:
            Authorised lease.
        marker:
            Verified signed evidence.

    Raises:
        HTTPException:
            If a different identity or mode owns the slot.
    """
    slot = next(
        (item for item in marker["leases"] if item["lease_id"] == lease["lease_id"]),
        None,
    )
    if (
        not slot
        or slot["contributor"].lower() != lease["contributor"].lower()
        or slot.get("ownership_mode", "assignee")
        != lease.get("ownership_mode", "assignee")
    ):
        raise HTTPException(
            status_code=409, detail="GitHub release ownership fence lost."
        )


def _verified(issue: GithubIssue) -> markers.VolunteerMarker | None:
    """Accept genuinely absent evidence but reject malformed or unsigned markers.

    Args:
        issue:
            Observed issue.

    Returns:
        Verified evidence or no marker.

    Raises:
        HTTPException:
            If marker syntax exists without valid signed evidence.
    """
    marker = markers.parse_volunteer_marker(body=issue["body"])
    if (marker is None and markers.has_volunteer_marker(body=issue["body"])) or (
        marker is not None
        and not markers.verify_volunteer_marker(
            issue_number=issue["number"], marker=marker
        )
    ):
        raise HTTPException(
            status_code=409,
            detail="GitHub ownership marker is missing, unsigned, or malformed.",
        )
    return marker
