"""Tests for the volunteer worker queue interlock."""

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest

from leaderboards.queue_markers import append_community_marker
from src.scripts import process_evaluation_queue


def test_claim_fails_closed_without_coordinator_mutex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared queue cannot silently fall back to an uncoordinated claim."""
    monkeypatch.delenv("VOLUNTEER_COORDINATOR_URL", raising=False)
    monkeypatch.delenv("WORKER_COORDINATOR_SECRET", raising=False)
    monkeypatch.delenv("VOLUNTEER_COORDINATOR_STANDALONE", raising=False)

    with pytest.raises(RuntimeError, match="coordinator URL and secret"):
        with process_evaluation_queue._coordinator_issue_lock(number=1):
            pass


def test_claim_recheck_excludes_issue_that_gains_community_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broker lease added after listing prevents a stale claim."""
    body = append_community_marker(
        body="request", owner="community", submission="submitted"
    )
    monkeypatch.setattr(
        process_evaluation_queue,
        "gh_request",
        lambda path: {"state": "open", "assignees": [], "body": body},
    )

    assert not process_evaluation_queue.issue_is_still_claimable(number=9)


def test_coordinator_lock_is_renewed_and_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held local lock renews its token and fences its release."""
    monkeypatch.setenv("VOLUNTEER_COORDINATOR_URL", "https://broker.test")
    monkeypatch.setenv("WORKER_COORDINATOR_SECRET", "secret")
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
        del number, secret
        calls.append((operation, token))
        return token or "lock-token"

    monkeypatch.setattr(process_evaluation_queue, "_coordinator_request", request)
    with process_evaluation_queue._coordinator_issue_lock(number=1):
        time.sleep(0.03)

    assert calls[0] == ("release", None)
    assert ("renew", "lock-token") in calls
    assert calls[-1] == ("release", "lock-token")


@pytest.mark.parametrize(
    ("assignees", "expected"),        [([], True), ([{"login": "runner"}], False), ([{"login": "other"}], False)],
)
def test_claim_recheck_respects_github_assignee(
    monkeypatch: pytest.MonkeyPatch, assignees: list[dict[str, str]], expected: bool
) -> None:
    """Claim rechecks accept only an unassigned or solely local issue."""
    monkeypatch.setattr(
        process_evaluation_queue,
        "gh_request",
        lambda path: {"state": "open", "assignees": assignees, "body": ""},
    )

    assert (            process_evaluation_queue.issue_is_still_claimable(number=9, assignee="runner")
        is expected
    )


def test_queue_candidates_exclude_active_community_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate filtering does not trust an issue with a broker lease."""
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


def _issue(number: int, body: str = "") -> dict[str, object]:
    """Build the minimal issue object used by candidate selection.

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


def test_queue_candidates_respect_assignee_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only unassigned issues enter a fresh local queue claim."""
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


def test_local_work_is_fenced_after_manual_reassignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement assignee prevents publishing results from old local work."""
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


def test_local_queue_code_has_no_coordinator_login_setting() -> None:
    """The local queue derives identity from its authenticated token."""
    paths = [Path("src/scripts/process_evaluation_queue.py")]
    paths.extend(Path("src/leaderboards").glob("queue_*.py"))

    assert all("WORKER_COORDINATOR_LOGIN" not in path.read_text() for path in paths)


def test_queue_candidates_paginate_past_first_hundred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An issue on page two is considered when page one is full."""
    pages = {1: [_issue(number) for number in range(1, 101)], 2: [_issue(101)]}
    requested_pages: list[int] = []

    def fake_request(path: str, *, params: dict[str, str]) -> list[dict[str, object]]:
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


def test_reclaim_keeps_accepted_marker_before_results_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accepted broker decision protects an assigned issue before labelling."""
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
        del vm_id, assignee
        released.append(number)
        return True

    monkeypatch.setattr(process_evaluation_queue, "release_issue_if_owned", release)

    process_evaluation_queue.reclaim_orphaned_issues(
        assignee="runner", vm_id="local-vm"
    )

    assert released == []
