"""Recover expired signed volunteer ownership safely."""

from __future__ import annotations

import math
import time
import typing as t

from upstash_redis.asyncio import Redis

from .github import GitHub, GithubIssue, Reconciliation
from .markers import (
    VolunteerMarker,
    parse_volunteer_marker,
    replace_volunteer_marker,
    sign_volunteer_marker,
    timestamp,
    verify_volunteer_marker,
    volunteer_assignees_match,
)
from .state import (
    IssueMutex,
    acquire_renewable_issue_mutex,
    get_lease_for_issue,
    reclaim_expired_lease,
)


class Recovery(t.TypedDict):
    """Updated issue and signed marker after lease recovery."""

    issue: GithubIssue
    marker: VolunteerMarker


async def recover_queue_ownership(
    redis: Redis, github: GitHub, issue_number: int
) -> Reconciliation | None:
    """Recover ownership for an open queue issue under its renewable mutex.

    Args:
        redis: Authoritative asynchronous Redis client.
        github: GitHub adapter.
        issue_number: Queue issue number.

    Returns:
        Reconciliation result, or None when recovery cannot be established.

    """
    mutex = await acquire_renewable_issue_mutex(redis=redis, issue=issue_number)
    if mutex is None:
        return None
    try:
        issue = await github.get_issue(number=issue_number)
        marker = parse_volunteer_marker(issue.get("body"))
        if (
            issue["state"] != "open"
            or marker is None
            or not verify_volunteer_marker(issue_number, marker)
        ):
            return None
        recovered = await recover_expired_ownership(redis, github, issue, marker, mutex)
        if recovered is None:
            return None
        return await github.reconcile_public_assignments(
            redis=redis,
            issue=recovered["issue"],
            marker=recovered["marker"],
            assert_owned=mutex.assert_owned,
        )
    finally:
        await mutex.release()


async def recover_expired_ownership(
    redis: Redis,
    github: GitHub,
    issue: GithubIssue,
    marker: VolunteerMarker,
    mutex: IssueMutex,
) -> Recovery | None:
    """Recover expired signed leases while preserving ownership evidence.

    Args:
        redis: Authoritative asynchronous Redis client.
        github: GitHub adapter used for issue mutations and reads.
        issue: Initially observed issue.
        marker: Initially verified volunteer marker.
        mutex: Renewable issue mutex fencing mutations.

    Returns:
        Updated issue and marker, or None when ownership is ambiguous.

    """
    current = issue
    current_marker = marker
    while True:
        now = time.time() * 1000
        expired = [
            lease
            for lease in current_marker["leases"]
            if timestamp(value=lease["expires_at"]) <= now
        ]
        if not expired:
            return {"issue": current, "marker": current_marker}
        lease = expired[0]
        contributor = lease["contributor"].lower()
        redis_lease = await get_lease_for_issue(
            redis=redis, issue=current["number"], language=lease["language"]
        )
        if redis_lease and (
            redis_lease["lease_id"] != lease["lease_id"]
            or redis_lease.get("contributor", "").lower() != contributor
            or not math.isfinite(timestamp(value=redis_lease["expires_at"]))
            or timestamp(value=redis_lease["expires_at"]) > time.time() * 1000
        ):
            return None
        if not volunteer_assignees_match(
            current["assignees"], current_marker, time.time() * 1000, True, True
        ):
            return None
        retained = any(
            other["lease_id"] != lease["lease_id"]
            and other.get("ownership_mode") != "public"
            and other["contributor"].lower() == contributor
            for other in current_marker["leases"]
        ) or any(
            submission["status"] in {"submitted", "accepted"}
            and submission.get("ownership_mode") != "public"
            and submission["verified_contributor"].lower() == contributor
            for submission in current_marker.get("submissions", [])
        )
        assigned = next(
            (
                item
                for item in current["assignees"]
                if item["login"].lower() == contributor
            ),
            None,
        )
        if lease.get("ownership_mode") != "public" and assigned and not retained:
            await mutex.assert_owned()
            try:
                await github.unassign_issue(
                    number=current["number"], login=assigned["login"]
                )
            except Exception:
                observed = await github.get_issue(number=current["number"])
                if any(
                    item["login"].lower() == contributor
                    for item in observed["assignees"]
                ):
                    raise
                current = observed
            current = await github.get_issue(number=current["number"])
            current_marker = parse_volunteer_marker(current.get("body"))
            if current_marker is None or not verify_volunteer_marker(
                current["number"], current_marker
            ):
                return None
            if any(
                item["lease_id"] == lease["lease_id"]
                for item in current_marker["leases"]
            ) and any(
                item["login"].lower() == contributor for item in current["assignees"]
            ):
                return None
            continue
        await mutex.assert_owned()
        status = current_marker["submission"]
        next_marker = sign_volunteer_marker(
            current["number"],
            {
                **current_marker,
                "submission": status
                if status in {"accepted", "rejected"}
                else "submitted"
                if current_marker.get("submissions")
                else "active",
                "leases": [
                    item
                    for item in current_marker["leases"]
                    if item["lease_id"] != lease["lease_id"]
                ],
            },
        )
        try:
            await github.patch_issue(
                number=current["number"],
                body=replace_volunteer_marker(current.get("body") or "", next_marker),
            )
        except Exception:
            observed = await github.get_issue(number=current["number"])
            observed_marker = parse_volunteer_marker(observed.get("body"))
            if (
                observed_marker is None
                or any(
                    item["lease_id"] == lease["lease_id"]
                    for item in observed_marker["leases"]
                )
                or not verify_volunteer_marker(observed["number"], observed_marker)
            ):
                raise
        current = await github.get_issue(number=current["number"])
        current_marker = parse_volunteer_marker(current.get("body"))
        if (
            current_marker is None
            or not verify_volunteer_marker(current["number"], current_marker)
            or any(
                item["lease_id"] == lease["lease_id"]
                for item in current_marker["leases"]
            )
        ):
            return None
        reconciled = await github.reconcile_public_assignments(
            redis=redis,
            issue=current,
            marker=current_marker,
            assert_owned=mutex.assert_owned,
        )
        current, current_marker = reconciled["issue"], reconciled["marker"]
        after = await get_lease_for_issue(
            redis=redis, issue=current["number"], language=lease["language"]
        )
        if after:
            expiry = timestamp(value=after["expires_at"])
            if (
                after["lease_id"] != lease["lease_id"]
                or not math.isfinite(expiry)
                or expiry > time.time() * 1000
            ):
                return None
            await mutex.assert_owned()
            if not await reclaim_expired_lease(
                redis=redis, lease=after
            ) and await get_lease_for_issue(
                redis=redis, issue=current["number"], language=lease["language"]
            ):
                return None
