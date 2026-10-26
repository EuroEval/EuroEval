"""Tests for the volunteer worker queue interlock."""

import base64
import hashlib
import hmac
import json
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from leaderboards.queue_markers import append_community_marker
from src.scripts import process_evaluation_queue


def test_claim_fails_closed_without_coordinator_mutex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared queue cannot silently fall back to an uncoordinated claim.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.delenv("VOLUNTEER_COORDINATOR_URL", raising=False)
    monkeypatch.delenv("WORKER_COORDINATOR_SECRET", raising=False)
    monkeypatch.delenv("VOLUNTEER_COORDINATOR_STANDALONE", raising=False)

    with pytest.raises(RuntimeError, match="coordinator URL.*secret"):
        with process_evaluation_queue._coordinator_issue_lock(number=1):
            pass


def test_claim_fails_closed_without_marker_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coordinator host must verify signed broker markers locally.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.setenv("VOLUNTEER_COORDINATOR_URL", "https://broker.test")
    monkeypatch.setenv("WORKER_COORDINATOR_SECRET", "coordinator-secret")
    monkeypatch.delenv("VOLUNTEER_MARKER_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="VOLUNTEER_MARKER_SECRET"):
        with process_evaluation_queue._coordinator_issue_lock(number=1):
            pass


def test_claim_recheck_excludes_issue_that_gains_community_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broker lease added after listing prevents a stale claim.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    body = append_community_marker(
        body="request", owner="community", submission="submitted"
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "gh_request",
        lambda path: {"state": "open", "assignees": [], "body": body},
    )

    assert not process_evaluation_queue.issue_is_still_claimable(number=9)


def test_claim_recheck_rejects_public_lease_added_after_candidate_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim-time fetch fences a broker lease racing candidate discovery.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.setenv("VOLUNTEER_MARKER_SECRET", "test-marker-secret")
    current = _issue(5, _signed_public_marker(issue_number=5, language="el"))
    current["state"] = "open"
    monkeypatch.setattr(process_evaluation_queue, "gh_request", lambda path: current)

    assert not process_evaluation_queue.issue_is_still_claimable(number=5)


def _issue(number: int, body: str = "") -> dict[str, object]:
    """Build the minimal issue object used by candidate selection.

    Args:
        number:
            Number supplied to the test.
        body (optional):
            Body supplied to the test. Defaults to ''.

    Returns:
        A minimal GitHub issue dictionary.
    """
    return {
        "number": number,
        "title": f"Evaluate model-{number}",
        "body": body,
        "labels": [],
        "created_at": "2026-01-01T00:00:00Z",
    }


def _signed_public_marker(
    *, issue_number: int, language: str, status: str = "active"
) -> str:
    """Build valid issue-bound public marker evidence for queue tests.

    Args:
        issue_number:
            Issue number supplied to the test.
        language:
            Language supplied to the test.
        status (optional):
            Status supplied to the test. Defaults to 'active'.

    Returns:
        A serialized issue-bound marker.
    """
    secret = "test-marker-secret"
    payload = {
        "protocol_version": "volunteer-worker/v1",
        "coordinator": "community",
        "submission": status,
        "leases": [],
        "submissions": [],
    }
    if status == "active":
        payload["leases"] = [
            {
                "lease_id": "lease-1",
                "language": language,
                "worker": "worker-1",
                "contributor": "volunteer",
                "expires_at": "2999-01-01T00:00:00+00:00",
                "ownership_mode": "public",
            }
        ]
    else:
        payload["submissions"] = [
            {
                "submission_id": "submission-1",
                "language": language,
                "manifest_path": "manifest.json",
                "submitted_at": "2026-01-01T00:00:00Z",
                "verified_contributor": "volunteer",
                "result_count": 1,
                "status": status,
                "ownership_mode": "public",
            }
        ]
    unsigned = {
        "domain": "euroeval-volunteer-marker",
        "version": 1,
        "issue_number": issue_number,
        "marker": payload,
    }
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    payload["signature"] = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode(), encoded, hashlib.sha256).digest()
        )
        .decode()
        .rstrip("=")
    )
    body = "request\n\n<!-- euroeval-volunteer-worker:v1 "
    return body + json.dumps(payload, separators=(",", ":")) + " -->\n"


@pytest.mark.parametrize(
    ("assignees", "expected"),
    [([], True), ([{"login": "runner"}], False), ([{"login": "other"}], False)],
)
def test_claim_recheck_respects_github_assignee(
    monkeypatch: pytest.MonkeyPatch, assignees: list[dict[str, str]], expected: bool
) -> None:
    """Claim rechecks accept only an unassigned or solely local issue.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        assignees:
            Assignees supplied to the test.
        expected:
            Expected value for this test case.
    """
    monkeypatch.setattr(
        process_evaluation_queue,
        "gh_request",
        lambda path: {"state": "open", "assignees": assignees, "body": ""},
    )

    assert (
        process_evaluation_queue.issue_is_still_claimable(number=9, assignee="runner")
        is expected
    )


def test_coordinator_lock_is_renewed_and_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held local lock renews its token and fences its release.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.setenv("VOLUNTEER_COORDINATOR_URL", "https://broker.test")
    monkeypatch.setenv("WORKER_COORDINATOR_SECRET", "secret")
    monkeypatch.setenv("VOLUNTEER_MARKER_SECRET", "marker-secret")
    monkeypatch.setattr(process_evaluation_queue, "COORDINATOR_RENEW_SECONDS", 0.01)
    calls: list[tuple[str, str | None]] = []

    def request(
        url: str,
        *,
        number: int,
        secret: str,
        token: str | None = None,
        operation: str = "release",
    ) -> str:
        """Request for this test.

        Args:
            url:
                Url supplied to the test.
            number:
                Number supplied to the test.
            secret:
                Secret supplied to the test.
            token (optional):
                Token supplied to the test. Defaults to None.
            operation (optional):
                Operation supplied to the test. Defaults to 'release'.

        Returns:
            The value returned by the test helper.
        """
        del number, secret
        calls.append((operation, token))
        return token or "lock-token"

    monkeypatch.setattr(process_evaluation_queue, "_coordinator_request", request)
    with process_evaluation_queue._coordinator_issue_lock(number=1):
        time.sleep(0.03)

    assert calls[0] == ("release", None)
    assert ("renew", "lock-token") in calls
    assert calls[-1] == ("release", "lock-token")


@pytest.mark.parametrize("status", ["active", "submitted"])
def test_local_claim_rechecks_cross_issue_public_work(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """A new public reservation on another issue fences a stale local candidate.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        status:
            Status supplied to the test.
    """
    monkeypatch.setenv("VOLUNTEER_MARKER_SECRET", "test-marker-secret")
    issue = _issue(1)
    other = _issue(
        2, _signed_public_marker(issue_number=2, language="el", status=status)
    )
    other["title"] = issue["title"]
    monkeypatch.setattr(
        process_evaluation_queue,
        "_list_queue_issues",
        lambda assignee: [other] if assignee == "none" else [],
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "extract_model_id",
        lambda title, body: title.removeprefix("Evaluate "),
    )
    coordinator = SimpleNamespace(ensure_healthy=lambda: None)
    monkeypatch.setattr(
        process_evaluation_queue,
        "_coordinator_issue_lock",
        lambda number: nullcontext(coordinator),
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "cached_model_summary",
        lambda model_id: {"gated": False},
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "set_vm_marker",
        lambda **kwargs: pytest.fail("local claim bypassed public reservation"),
    )

    process_evaluation_queue.process_issue(
        issue=issue,
        model_id="model-1",
        groups=["Greek"],
        assignee="runner",
        vm_id="vm",
        gpu_memory_utilization=None,
    )


def test_local_queue_code_has_no_coordinator_login_setting() -> None:
    """The local queue derives identity from its authenticated token."""
    paths = [Path("src/scripts/process_evaluation_queue.py")]
    paths.extend(Path("src/leaderboards").glob("queue_*.py"))

    assert all("WORKER_COORDINATOR_LOGIN" not in path.read_text() for path in paths)


def test_local_work_is_fenced_after_manual_reassignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement assignee prevents publishing results from old local work.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    issue = {
        "state": "open",
        "body": "<!-- vm-id: local-vm -->",
        "assignees": [{"login": "manual-owner"}],
    }
    uploaded: list[list[str]] = []
    monkeypatch.setattr(process_evaluation_queue, "gh_request", lambda path: issue)
    monkeypatch.setattr(
        process_evaluation_queue,
        "upload_results_to_hf_bucket",
        lambda lines, model_id: uploaded.append(lines) or True,
    )

    result = process_evaluation_queue._upload_if_owned(
        number=1,
        vm_id="local-vm",
        assignee="runner",
        lines=["result"],
        model_id="model",
        language="el",
    )

    assert result is None
    assert uploaded == []


def test_public_lease_language_blocks_local_candidate_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signed public lease protects its language across unassigned issues.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.setenv("VOLUNTEER_MARKER_SECRET", "test-marker-secret")
    issues = [
        _issue(1, _signed_public_marker(issue_number=1, language="el")),
        _issue(2),
        _issue(3, "english"),
    ]
    assert process_evaluation_queue.public_marker_protected_languages(
        str(issues[0]["body"]), issue_number=1
    ) == {"el"}
    monkeypatch.setattr(
        process_evaluation_queue,
        "_list_queue_issues",
        lambda assignee: issues if assignee == "none" else [],
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "extract_model_id",
        lambda title, body: title.removeprefix("Evaluate "),
    )
    west_germanic = next(
        group
        for group, codes in process_evaluation_queue.LANGUAGE_GROUP_CODES.items()
        if "en" in codes
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "extract_language_groups",
        lambda body: [west_germanic] if body == "english" else ["Greek"],
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "cached_model_summary",
        lambda model_id: {"param_count": 1, "generative": True, "gated": False},
    )

    candidates = process_evaluation_queue._queue_candidates()

    assert [candidate[5]["number"] for candidate in candidates] == [3]


def test_queue_candidates_exclude_active_community_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate filtering does not trust an issue with a broker lease.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    marker_body = append_community_marker(
        body="request", owner="community", submission="active"
    ).replace("} -->", ',"signature":"signed"} -->')
    issues = [_issue(1), _issue(2, marker_body)]
    monkeypatch.setattr(
        process_evaluation_queue, "gh_request", lambda path, *, params: issues
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "extract_model_id",
        lambda title, body: title.removeprefix("Evaluate "),
    )
    monkeypatch.setattr(
        process_evaluation_queue, "extract_language_groups", lambda body: ["Greek"]
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "cached_model_summary",
        lambda model_id: {"param_count": 1, "generative": True, "gated": False},
    )

    candidates = process_evaluation_queue._queue_candidates()

    assert [candidate[5]["number"] for candidate in candidates] == [1]


def test_queue_candidates_paginate_past_first_hundred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An issue on page two is considered when page one is full.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    pages = {1: [_issue(number) for number in range(1, 101)], 2: [_issue(101)]}
    requested_pages: list[int] = []

    def fake_request(path: str, *, params: dict[str, str]) -> list[dict[str, object]]:
        """Fake request for this test.

        Args:
            path:
                Path supplied to the test.
            params:
                Params supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        requested_pages.append(int(params["page"]))
        return pages[int(params["page"])]

    monkeypatch.setattr(process_evaluation_queue, "gh_request", fake_request)
    monkeypatch.setattr(
        process_evaluation_queue,
        "extract_model_id",
        lambda title, body: title.removeprefix("Evaluate "),
    )
    monkeypatch.setattr(
        process_evaluation_queue, "extract_language_groups", lambda body: ["Greek"]
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "cached_model_summary",
        lambda model_id: {"param_count": 1, "generative": True, "gated": False},
    )

    candidates = process_evaluation_queue._queue_candidates()

    assert requested_pages == [1, 2]
    assert {candidate[5]["number"] for candidate in candidates} == set(range(1, 102))


def test_queue_candidates_respect_assignee_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only unassigned issues enter a fresh local queue claim.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    issues = [_issue(1), _issue(2), _issue(3)]
    issues[1]["assignees"] = [{"login": "runner"}]
    issues[2]["assignees"] = [{"login": "manual-owner"}]
    monkeypatch.setattr(
        process_evaluation_queue, "gh_request", lambda path, *, params: issues
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "extract_model_id",
        lambda title, body: title.removeprefix("Evaluate "),
    )
    monkeypatch.setattr(
        process_evaluation_queue, "extract_language_groups", lambda body: ["Greek"]
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "cached_model_summary",
        lambda model_id: {"param_count": 1, "generative": True, "gated": False},
    )

    candidates = process_evaluation_queue._queue_candidates(assignee="runner")

    assert [candidate[5]["number"] for candidate in candidates] == [1]


def test_reclaim_keeps_accepted_marker_before_results_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accepted broker decision protects an assigned issue before labelling.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    secret = "marker-secret"
    marker = {
        "protocol_version": "volunteer-worker/v1",
        "coordinator": "coordinator",
        "submission": "accepted",
        "leases": [],
        "submissions": [
            {
                "submission_id": "submission-1",
                "language": "el",
                "manifest_path": "volunteer/manifests/submission-1.json",
                "submitted_at": "2026-09-06T10:00:00Z",
                "verified_contributor": "alice",
                "result_count": 1,
                "status": "accepted",
            }
        ],
        "completed_languages": ["el"],
    }
    payload = json.dumps(
        {
            "domain": "euroeval-volunteer-marker",
            "version": 1,
            "issue_number": 7,
            "marker": marker,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    marker["signature"] = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode(), payload, hashlib.sha256).digest()
        )
        .decode()
        .rstrip("=")
    )
    body = (
        f"request\n<!-- euroeval-volunteer-worker:v1 {json.dumps(marker)} -->\n"
        "<!-- vm-id: local-vm -->"
    )
    released: list[int] = []
    monkeypatch.setenv("VOLUNTEER_MARKER_SECRET", secret)
    monkeypatch.setattr(
        process_evaluation_queue,
        "_list_queue_issues",
        lambda *, assignee: [_issue(7, body)],
    )

    def release(*, number: int, vm_id: str, assignee: str) -> bool:
        """Release for this test.

        Args:
            number:
                Number supplied to the test.
            vm_id:
                Vm id supplied to the test.
            assignee:
                Assignee supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        del vm_id, assignee
        released.append(number)
        return True

    monkeypatch.setattr(process_evaluation_queue, "release_issue_if_owned", release)

    process_evaluation_queue.reclaim_orphaned_issues(
        assignee="runner", vm_id="local-vm"
    )

    assert released == []


def test_unsigned_public_marker_fails_closed_for_unassigned_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsigned public ownership cannot authorize an unassigned queue candidate.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    payload = {
        "protocol_version": "volunteer-worker/v1",
        "coordinator": "community",
        "submission": "active",
        "leases": [
            {
                "lease_id": "lease-1",
                "language": "el",
                "worker": "worker-1",
                "contributor": "volunteer",
                "expires_at": "2999-01-01T00:00:00+00:00",
                "ownership_mode": "public",
            }
        ],
    }
    unsigned_body = (
        "request\\n\\n<!-- euroeval-volunteer-worker:v1 " + json.dumps(payload) + " -->"
    )
    issue = _issue(1, unsigned_body)
    monkeypatch.setattr(
        process_evaluation_queue,
        "_list_queue_issues",
        lambda assignee: [issue] if assignee == "none" else [],
    )

    assert process_evaluation_queue._queue_candidates() == []
