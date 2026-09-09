"""Tests for the volunteer worker queue interlock."""

import pytest

from leaderboards.queue_markers import append_community_marker
from src.scripts import process_evaluation_queue


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


def test_queue_candidates_exclude_active_community_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate filtering does not trust an issue with a broker lease."""
    marker_body = append_community_marker(
        body="request", owner="community", submission="active"
    )
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
