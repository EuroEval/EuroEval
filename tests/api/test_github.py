"""Public outcomes for GitHub lease ownership and assignment reconciliation."""

import json

import httpx
import pytest
from fastapi import HTTPException
from upstash_redis.asyncio import Redis

from euroeval_api.github import GitHub, GithubIssue, require_lease_ownership
from euroeval_api.markers import (
    MarkerLease,
    VolunteerMarker,
    sign_volunteer_marker,
    verify_volunteer_marker,
)
from euroeval_api.state import issue_lease_key


def test_lease_ownership_distinguishes_public_and_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow public ownership without assignment and enforce legacy assignment.

    Args:
        monkeypatch:
            Supplies a fabricated signing secret.
    """
    monkeypatch.setenv("VOLUNTEER_MARKER_SECRET", "fabricated-marker-secret")
    lease: MarkerLease = {
        "lease_id": "lease-1",
        "language": "da",
        "worker": "worker-1",
        "contributor": "owner",
        "expires_at": "2099-01-01T00:00:00Z",
        "ownership_mode": "public",
    }
    base: VolunteerMarker = {
        "protocol_version": "volunteer-worker/v1",
        "coordinator": "broker",
        "submission": "active",
        "leases": [lease],
    }
    signed = sign_volunteer_marker(issue_number=42, marker=base)
    issue: GithubIssue = {
        "number": 42,
        "title": "Request",
        "body": "<!-- euroeval-volunteer-worker:v1 " + json.dumps(signed) + " -->",
        "state": "open",
        "assignees": [],
    }
    require_lease_ownership(issue=issue, lease=lease)
    legacy_lease: MarkerLease = {**lease, "ownership_mode": "assignee"}
    legacy_marker = sign_volunteer_marker(
        issue_number=42, marker={**base, "leases": [legacy_lease]}
    )
    legacy_issue: GithubIssue = {
        **issue,
        "body": "<!-- euroeval-volunteer-worker:v1 "
        + json.dumps(legacy_marker)
        + " -->",
    }
    with pytest.raises(HTTPException):
        require_lease_ownership(issue=legacy_issue, lease=legacy_lease)
    for altered in (
        {**signed, "signature": "invalid-signature"},
        {**signed, "coordinator": "different-broker"},
    ):
        tampered: GithubIssue = {
            **issue,
            "body": "<!-- euroeval-volunteer-worker:v1 " + json.dumps(altered) + " -->",
        }
        with pytest.raises(HTTPException):
            require_lease_ownership(issue=tampered, lease=lease)


@pytest.mark.anyio
async def test_reconciliation_protects_replacement_and_manual_assignees(
    monkeypatch: pytest.MonkeyPatch, upstash_client: Redis
) -> None:
    """Retain stale ownership when Redis has a different live lease owner.

    Args:
        monkeypatch:
            Supplies fabricated GitHub and marker credentials.
        upstash_client:
            Real Redis-backed Upstash client fixture.
    """
    redis = upstash_client
    monkeypatch.setenv("VOLUNTEER_MARKER_SECRET", "fabricated-marker-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "fabricated-github-token")
    old_lease: MarkerLease = {
        "lease_id": "old-lease",
        "language": "da",
        "worker": "old-worker",
        "contributor": "old-owner",
        "expires_at": "2000-01-01T00:00:00Z",
        "ownership_mode": "public",
    }
    replacement: MarkerLease = {
        **old_lease,
        "lease_id": "new-lease",
        "worker": "new-worker",
        "contributor": "new-owner",
        "expires_at": "2099-01-01T00:00:00Z",
    }
    await redis.set(issue_lease_key(issue=42, language="da"), json.dumps(replacement))
    marker: VolunteerMarker = {
        "protocol_version": "volunteer-worker/v1",
        "coordinator": "broker",
        "submission": "active",
        "leases": [old_lease],
        "public_assignees": ["old-owner"],
    }
    signed = sign_volunteer_marker(issue_number=42, marker=marker)
    body = (
        "Issue body\n<!-- euroeval-volunteer-worker:v1 " + json.dumps(signed) + " -->"
    )
    issue: GithubIssue = {
        "number": 42,
        "title": "Evaluation request",
        "body": body,
        "state": "open",
        "assignees": [{"login": "old-owner"}, {"login": "manual-reviewer"}],
    }
    requests: list[tuple[str, str]] = []

    async def github_handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        if request.method == "DELETE":
            return httpx.Response(status_code=500)
        if request.method in {"GET", "PATCH"}:
            return httpx.Response(status_code=200, json=issue)
        return httpx.Response(status_code=200, json={})

    async def assert_owned() -> None:
        return None

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(github_handler)
    ) as client:
        result = await GitHub(http=client).reconcile_public_assignments(
            redis=redis, issue=issue, marker=marker, assert_owned=assert_owned
        )
    assert result["pending_removals"] == ["old-owner"]
    assert result["issue"]["assignees"] == [
        {"login": "old-owner"},
        {"login": "manual-reviewer"},
    ]
    assert result["marker"]["public_assignees"] == ["old-owner"]
    assert verify_volunteer_marker(issue_number=42, marker=result["marker"])
    assert not any(method == "POST" for method, _ in requests)
