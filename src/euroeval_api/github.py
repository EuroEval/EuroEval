"""GitHub ownership and best-effort public assignment projection."""

from __future__ import annotations

import collections.abc as c
import math
import time
import typing as t
from urllib.parse import quote

import httpx
from fastapi import HTTPException
from upstash_redis.asyncio import Redis

from .dependencies import fetch_with_retry, require_env
from .markers import (
    VolunteerMarker,
    parse_volunteer_marker,
    replace_volunteer_marker,
    sign_volunteer_marker,
    timestamp,
    verify_volunteer_marker,
)
from .state import get_lease_for_issue

REPOSITORY = "EuroEval/EuroEval"
REQUEST_LABEL = "model evaluation request"


class GithubIssue(t.TypedDict):
    """The issue fields needed by the broker, retaining the original JSON object."""

    number: int
    title: str
    body: str | None
    state: str
    assignees: t.NotRequired[list[dict[str, str]]]
    labels: t.NotRequired[list[dict[str, str]]]
    pull_request: t.NotRequired[object]


def has_issue_assignee(issue: GithubIssue, login: str) -> bool:
    """Check assignment case-insensitively without treating it as public ownership.

    Args:
        issue:
            Current issue.
        login:
            Contributor login.

    Returns:
        Whether the contributor is assigned.
    """
    return any(
        item["login"].lower() == login.lower()
        for item in issue.get("assignees", []) or []
    )


class Reconciliation(t.TypedDict):
    """Observed ownership projection and cleanup that still requires a retry."""

    issue: GithubIssue
    marker: VolunteerMarker
    pending_removals: list[str]


class GitHub:
    """Share authenticated reads and mutations without replaying ambiguous writes."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        """Use the application's outbound connection pool.

        Args:
            http:
                Shared outbound HTTP client.
        """
        self.http = http

    async def add_issue_label(self, number: int, label: str) -> None:
        """Add the requested workflow label.

        Args:
            number:
                Issue number.
            label:
                Label to add.
        """
        await self.request(
            path=f"/repos/{REPOSITORY}/issues/{number}/labels",
            method="POST",
            body={"labels": [label]},
        )

    async def request(
        self, path: str, method: str = "GET", body: object | None = None
    ) -> object:
        """Call GitHub with the broker's configured authority and safe-read retries.

        Args:
            path:
                GitHub API path.
            method (optional):
                HTTP method, defaulting to GET.
            body (optional):
                JSON payload, defaulting to none.

        Returns:
            Decoded response, or none for an empty success.

        Raises:
            HTTPException:
                If GitHub rejects or cannot complete the request.
        """
        response = await fetch_with_retry(
            http=self.http,
            url=f"https://api.github.com{path}",
            method=method,
            headers={
                "accept": "application/vnd.github+json",
                "x-github-api-version": "2022-11-28",
                "authorization": f"Bearer {require_env(name='GITHUB_TOKEN')}",
            },
            body=body,
        )
        if not response.is_success:
            raise HTTPException(
                status_code=404 if response.status_code == 404 else 502,
                detail=(
                    f"GitHub returned HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                ),
            )
        return None if response.status_code == 204 else response.json()

    async def assert_assignable(self, login: str) -> None:
        """Distinguish ineligible contributors from unavailable eligibility checks.

        Args:
            login:
                Contributor to check.

        Raises:
            HTTPException:
                If the contributor is ineligible or cannot be verified.
        """
        try:
            await self.request(
                path=f"/repos/{REPOSITORY}/assignees/{quote(login, safe='')}"
            )
        except HTTPException as error:
            if error.status_code == 404:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": (
                            "Your GitHub login cannot be assigned to "
                            "evaluation issues; check repository access "
                            "and try again."
                        ),
                        "code": "github_login_not_assignable",
                    },
                ) from error
            raise HTTPException(
                status_code=502,
                detail={
                    "error": (
                        "Unable to verify whether your GitHub login "
                        "can be assigned; try again later."
                    ),
                    "code": "assignee_eligibility_unavailable",
                },
            ) from error

    async def comment_issue(self, number: int, body: str) -> None:
        """Post once after the caller establishes ownership and deduplication.

        Args:
            number:
                Issue number.
            body:
                Comment body.
        """
        await self.request(
            path=f"/repos/{REPOSITORY}/issues/{number}/comments",
            method="POST",
            body={"body": body},
        )

    async def issue_comments(self, number: int) -> list[dict[str, object]]:
        """Read the existing bounded comment projection.

        Args:
            number:
                Issue number.

        Returns:
            First comment page.
        """
        return t.cast(
            list[dict[str, object]],
            await self.request(
                path=f"/repos/{REPOSITORY}/issues/{number}/comments?per_page=100"
            ),
        )

    async def list_open_issues(self) -> list[GithubIssue]:
        """List the bounded evaluation queue without treating pull requests as work.

        Returns:
            Up to ten pages of open evaluation requests.
        """
        issues: list[GithubIssue] = []
        label = quote(REQUEST_LABEL, safe="")
        for page in range(1, 11):
            batch = t.cast(
                list[GithubIssue],
                await self.request(
                    path=(
                        f"/repos/{REPOSITORY}/issues?state=open&labels={label}"
                        f"&per_page=100&page={page}"
                    )
                ),
            )
            issues.extend(issue for issue in batch if "pull_request" not in issue)
            if len(batch) < 100:
                break
        return issues

    async def reconcile_public_assignments(
        self,
        redis: Redis,
        issue: GithubIssue,
        marker: VolunteerMarker,
        assert_owned: c.Callable[[], c.Awaitable[None]],
    ) -> Reconciliation:
        """Project signed ownership without deleting legacy or uncertain assignments.

        Args:
            redis:
                Official client for authoritative lease renewals.
            issue:
                Current issue projection.
            marker:
                Signed ownership evidence.
            assert_owned:
                Required issue-mutex ownership fence.

        Returns:
            Observed issue, signed marker and removals still pending.
        """
        now = time.time()
        desired = {
            lease["contributor"].lower(): lease["contributor"]
            for lease in marker["leases"]
            if lease.get("ownership_mode") == "public"
            and timestamp(value=lease["expires_at"]) > now
        }
        protected = {
            lease["contributor"].lower()
            for lease in marker["leases"]
            if lease.get("ownership_mode") != "public"
        }
        protected.update(
            submission["verified_contributor"].lower()
            for submission in marker.get("submissions", [])
            if submission.get("ownership_mode") != "public"
            and submission["status"] in {"submitted", "accepted"}
        )
        tracked = {login.lower(): login for login in marker.get("public_assignees", [])}
        pending: list[str] = []
        for lease in marker["leases"]:
            if (
                lease.get("ownership_mode") != "public"
                or timestamp(value=lease["expires_at"]) > now
            ):
                continue
            login = lease["contributor"].lower()
            try:
                active = await get_lease_for_issue(
                    redis=redis, issue=issue["number"], language=lease["language"]
                )
                if not active or active.get("released"):
                    continue
                expiry = timestamp(value=active["expires_at"])
                if not math.isfinite(expiry):
                    protected.add(login)
                elif expiry > now:
                    protected.add(login)
                    contributor = active.get("contributor")
                    if (
                        active["lease_id"] == lease["lease_id"]
                        and isinstance(contributor, str)
                        and contributor.lower() == login
                    ):
                        desired[login] = lease["contributor"]
            except Exception:
                protected.add(login)
        current = issue
        for key, login in desired.items():
            if has_issue_assignee(issue=current, login=login):
                continue
            try:
                await assert_owned()
                await self.assign_issue(number=issue["number"], login=login)
                observed = await self.get_issue(number=issue["number"])
                if has_issue_assignee(issue=observed, login=login):
                    tracked[key] = login
                    current = observed
            except Exception:
                continue
        for key, login in list(tracked.items()):
            if key in desired or key in protected:
                continue
            if not has_issue_assignee(issue=current, login=login):
                del tracked[key]
                continue
            try:
                await assert_owned()
                await self.unassign_issue(number=issue["number"], login=login)
                observed = await self.get_issue(number=issue["number"])
                if not has_issue_assignee(issue=observed, login=login):
                    del tracked[key]
                else:
                    pending.append(login)
                current = observed
            except Exception:
                pending.append(login)
        next_logins = list(tracked.values())
        previous = marker.get("public_assignees", [])
        if len(next_logins) != len(previous) or any(
            login.lower() not in {item.lower() for item in previous}
            for login in next_logins
        ):
            await assert_owned()
            fresh = await self.get_issue(number=issue["number"])
            fresh_marker = parse_volunteer_marker(body=fresh["body"])
            if not fresh_marker or not verify_volunteer_marker(
                issue_number=issue["number"], marker=fresh_marker
            ):
                return {"issue": fresh, "marker": marker, "pending_removals": pending}
            signed = sign_volunteer_marker(
                issue_number=issue["number"],
                marker={**fresh_marker, "public_assignees": next_logins},
            )
            await assert_owned()
            await self.patch_issue(
                number=issue["number"],
                body=replace_volunteer_marker(body=fresh["body"] or "", marker=signed),
            )
            current = await self.get_issue(number=issue["number"])
            patched = parse_volunteer_marker(body=current["body"])
            if patched and verify_volunteer_marker(
                issue_number=issue["number"], marker=patched
            ):
                return {
                    "issue": current,
                    "marker": patched,
                    "pending_removals": pending,
                }
        return {"issue": current, "marker": marker, "pending_removals": pending}

    async def assign_issue(self, number: int, login: str) -> None:
        """Request an assignment without assuming GitHub actually applies it.

        Args:
            number:
                Issue number.
            login:
                Contributor to assign.
        """
        await self.request(
            path=f"/repos/{REPOSITORY}/issues/{number}/assignees",
            method="POST",
            body={"assignees": [login]},
        )

    async def get_issue(self, number: int) -> GithubIssue:
        """Fetch the current issue, rather than inferring ownership from a cache.

        Args:
            number:
                Issue number.

        Returns:
            Current issue JSON.
        """
        return t.cast(
            GithubIssue, await self.request(path=f"/repos/{REPOSITORY}/issues/{number}")
        )

    async def patch_issue(self, number: int, body: str) -> None:
        """Replace an issue body after the caller checks its ownership fence.

        Args:
            number:
                Issue number.
            body:
                Complete replacement body.
        """
        await self.request(
            path=f"/repos/{REPOSITORY}/issues/{number}",
            method="PATCH",
            body={"body": body},
        )

    async def unassign_issue(self, number: int, login: str) -> None:
        """Remove the explicitly tracked public assignment.

        Args:
            number:
                Issue number.
            login:
                Tracked contributor to remove.
        """
        await self.request(
            path=f"/repos/{REPOSITORY}/issues/{number}/assignees",
            method="DELETE",
            body={"assignees": [login]},
        )

    async def remove_issue_label(self, number: int, label: str) -> None:
        """Remove a workflow label, allowing an already absent label.

        Args:
            number:
                Issue number.
            label:
                Label to remove.

        Raises:
            HTTPException:
                If GitHub fails for a reason other than an already absent label.
        """
        encoded_label = quote(label, safe="")
        try:
            await self.request(
                path=f"/repos/{REPOSITORY}/issues/{number}/labels/{encoded_label}",
                method="DELETE",
            )
        except HTTPException as error:
            if error.status_code != 404:
                raise


def require_lease_ownership(issue: GithubIssue, lease: c.Mapping[str, object]) -> None:
    """Bind mutations to the signed lease, retaining public/legacy ownership modes.

    Args:
        issue:
            Current issue.
        lease:
            Authoritative persisted lease.

    Raises:
        HTTPException:
            If signed ownership or required legacy assignment was lost.
    """
    marker = parse_volunteer_marker(body=issue["body"])
    owned = (
        next(
            (
                entry
                for entry in marker["leases"]
                if entry["lease_id"] == lease["lease_id"]
            ),
            None,
        )
        if marker
        else None
    )
    contributor = lease.get("contributor")
    if (
        not marker
        or not owned
        or not verify_volunteer_marker(issue_number=issue["number"], marker=marker)
        or not isinstance(contributor, str)
        or owned["contributor"].lower() != contributor.lower()
        or owned.get("ownership_mode", "assignee")
        != lease.get("ownership_mode", "assignee")
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": (
                    "The signed issue ownership marker is missing,"
                    " invalid, or no longer owns this lease."
                ),
                "code": "lease_ownership_lost",
            },
        )
    if lease.get("ownership_mode") != "public":
        require_assignee(issue=issue, login=contributor)


def require_assignee(issue: GithubIssue, login: str) -> None:
    """Require legacy assignment ownership.

    Args:
        issue:
            Current issue.
        login:
            Lease contributor.

    Raises:
        HTTPException:
            If the required assignment disappeared.
    """
    if not has_issue_assignee(issue=issue, login=login):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "The lease contributor is no longer assigned to this issue.",
                "code": "lease_assignment_lost",
            },
        )
