"""Tests for queue issue-body ownership markers."""

import pytest

from leaderboards import queue_markers


def test_community_marker_parser_accepts_protocol_v1() -> None:
    """A canonical v1 marker parses into its ownership fields."""
    body = queue_markers.append_community_marker(
        body="request", owner="community", submission="running"
    )

    marker = queue_markers.parse_community_marker(body)

    assert marker == queue_markers.CommunityMarker(
        protocol_version=1, owner="community", submission="running"
    )
    assert queue_markers.issue_has_active_queue_ownership(body)


def test_community_marker_parser_rejects_malformed_and_unknown_versions() -> None:
    """Malformed and future markers must not accidentally claim an issue."""
    malformed = "request\n<!-- euroeval-community: v1 owner=community -->"
    unknown_version = (
        "request\n<!-- euroeval-community: v2 owner=community submission=running -->"
    )

    assert queue_markers.parse_community_marker(malformed) is None
    assert queue_markers.parse_community_marker(unknown_version) is None
    assert not queue_markers.issue_has_active_queue_ownership(unknown_version)


def test_release_does_not_touch_community_owned_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A community lease protects both its marker and the assignment."""
    body = queue_markers.append_community_marker(
        body="\n<!-- vm-id: local-vm -->", owner="community", submission="submitted"
    )
    patched: list[str] = []
    unassigned: list[int] = []
    monkeypatch.setattr(queue_markers, "fetch_issue_body", lambda number: body)
    monkeypatch.setattr(
        queue_markers, "patch_issue_body", lambda number, body: patched.append(body)
    )
    monkeypatch.setattr(
        queue_markers,
        "unassign_issue",
        lambda number, assignee: unassigned.append(number),
    )

    released = queue_markers.release_issue_if_owned(
        number=12, vm_id="local-vm", assignee="runner"
    )

    assert not released
    assert patched == []
    assert unassigned == []


def test_expired_coordinator_marker_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expired broker ownership does not strand the local queue."""
    body = (
        '<!-- euroeval-volunteer-worker:v1 {"protocol_version":"volunteer-worker/v1",'
        '"coordinator":"coordinator","submission":"active","leases":['
        '{"lease_id":"old","language":"da","worker":"w",'
        '"contributor":"c","expires_at":"2000-01-01T00:00:00Z"}]} -->'
    )
    patched: list[str] = []
    monkeypatch.setattr(queue_markers, "fetch_issue_body", lambda number: body)
    monkeypatch.setattr(
        queue_markers, "patch_issue_body", lambda number, body: patched.append(body)
    )

    assert not queue_markers.issue_has_active_queue_ownership(body)
    assert queue_markers.set_vm_marker(number=1, vm_id="local-vm")
    assert "euroeval-volunteer-worker" not in patched[0]
    assert "vm-id: local-vm" in patched[0]


def test_vm_marker_manipulators_leave_active_community_marker_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim and cleanup paths must both honour a broker lease."""
    body = queue_markers.append_community_marker(
        body="request", owner="coordinator", submission="pending"
    )
    patched: list[str] = []
    monkeypatch.setattr(queue_markers, "fetch_issue_body", lambda number: body)
    monkeypatch.setattr(
        queue_markers, "patch_issue_body", lambda number, body: patched.append(body)
    )

    assert not queue_markers.set_vm_marker(number=1, vm_id="local-vm")
    assert not queue_markers.vm_marker_matches(number=1, vm_id="local-vm")
    queue_markers.clear_vm_marker(number=1, vm_id="local-vm")
    assert patched == []
