"""Public outcomes for volunteer marker compatibility."""

from euroeval_api.markers import (
    parse_volunteer_marker,
    sign_volunteer_marker,
    verify_volunteer_marker,
    volunteer_assignees_match,
)

BASE = {
    "protocol_version": "volunteer-worker/v1",
    "coordinator": "broker",
    "submission": "active",
    "leases": [],
}


def test_persisted_typescript_signature_vector(monkeypatch):
    """Verify a persisted signature generated with the TypeScript canonical format."""
    monkeypatch.setenv("VOLUNTEER_MARKER_SECRET", "test-secret")
    marker = {**BASE, "signature": "_LRSScsT6dEomQXUC32DQDMOWhYbCrHk47hINbz6PHY"}
    assert verify_volunteer_marker(42, marker) is False
    signed = sign_volunteer_marker(42, BASE)
    assert signed["signature"] == "_LRSScsT6dEomQXUC32DQDMOWhYbCrHk47hINbz6PHY"


def test_parser_rejects_duplicate_and_invalid_markers():
    """Reject duplicate marker comments and malformed marker payloads."""
    good = '<!-- euroeval-volunteer-worker:v1 {"protocol_version":"volunteer-worker/v1","coordinator":"broker","submission":"active","leases":[]} -->'
    assert parse_volunteer_marker(good) is not None
    assert parse_volunteer_marker(good + "\n" + good) is None
    assert parse_volunteer_marker(good.replace('"leases":[]', '"leases":[{}]')) is None


def test_public_legacy_and_manual_assignee_rules():
    """Allow public ownership without assigned contributors, but reject unrelated assignees otherwise."""
    assert volunteer_assignees_match([{"login": "other"}], {**BASE, "leases": [{"lease_id": "l", "language": "da", "worker": "w", "contributor": "owner", "expires_at": "2099-01-01T00:00:00Z", "ownership_mode": "public"}]}, now=0)
    assert not volunteer_assignees_match([{"login": "manual"}], BASE, now=0)
    assert volunteer_assignees_match([{"login": "owner"}], {**BASE, "leases": [{"lease_id": "l", "language": "da", "worker": "w", "contributor": "owner", "expires_at": "2000-01-01T00:00:00Z"}]}, now=1, include_expired_leases=True, allow_expired_unassigned=True)
