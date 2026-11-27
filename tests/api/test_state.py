"""Public outcome tests for Redis-backed API state operations."""

from __future__ import annotations

import json
import typing as t

import pytest
from upstash_redis.asyncio import Redis

from euroeval_api.state import (
    LeaseRecord,
    delete_lease,
    get_lease_by_id,
    identity_key,
    issue_lease_key,
    lease_key,
    put_lease,
    reclaim_expired_lease,
    reserve_result_identity,
    result_reservations_key,
)


@pytest.mark.anyio
async def test_expired_reclaim_only_removes_matching_reservation(
    upstash_client: Redis,
) -> None:
    """Expired lease reclamation leaves unrelated reservation owners intact."""
    record = lease("expired")
    identity = "model/dataset"
    key = identity_key(identity)
    await upstash_client.set(
        lease_key("expired"), json.dumps(record, separators=(",", ":"))
    )
    await upstash_client.set(
        issue_lease_key(42, "da"), json.dumps(record, separators=(",", ":"))
    )
    await upstash_client.set(
        key, json.dumps({"lease_id": "other", "digest": "new", "status": "uploading"})
    )
    assert not await reclaim_expired_lease(redis=upstash_client, lease=record)
    assert await upstash_client.get(key) is not None


def lease(lease_id: str, issue: int = 42) -> LeaseRecord:
    """Create a lease record with an opaque extension field.

    Args:
        lease_id: Lease identifier.
        issue: GitHub issue number.

    Returns:
        A lease record suitable for persistence.
    """
    return t.cast(
        LeaseRecord,
        {
            "lease_id": lease_id,
            "issue_number": issue,
            "language": "da",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "opaque": {"kept": True},
        },
    )


@pytest.mark.anyio
async def test_lease_insert_fencing_and_released_tombstone(
    upstash_client: Redis,
) -> None:
    """Lease ownership is fenced and released records remain readable."""
    owner = lease("owner")
    foreign = lease("foreign")
    assert await put_lease(redis=upstash_client, lease=owner)
    assert not await put_lease(redis=upstash_client, lease=foreign)
    stored = await get_lease_by_id(redis=upstash_client, lease_id="owner")
    assert stored == owner

    await delete_lease(redis=upstash_client, lease=owner)
    tombstone = await get_lease_by_id(redis=upstash_client, lease_id="owner")
    assert tombstone is not None
    assert tombstone["lease_id"] == "owner"
    assert await put_lease(redis=upstash_client, lease=foreign)
    await delete_lease(redis=upstash_client, lease=owner)
    assert await get_lease_by_id(redis=upstash_client, lease_id="foreign") == foreign
    assert await upstash_client.get(issue_lease_key(42, "da")) == json.dumps(
        foreign, separators=(",", ":")
    )


@pytest.mark.anyio
async def test_reservations_distinguish_retry_duplicate_and_digest_conflict(
    upstash_client: Redis,
) -> None:
    """Reservations report retries, completed duplicates and byte conflicts."""
    key = identity_key("model/dataset")
    tracked = result_reservations_key("lease-a")
    reservation = json.dumps(
        {"status": "uploading", "lease_id": "lease-a", "digest": "abc"},
        separators=(",", ":"),
    )
    entry = json.dumps({"identity": "model/dataset", "digest": "abc"})
    assert (
        await reserve_result_identity(
            redis=upstash_client,
            key=key,
            reservation=reservation,
            digest="abc",
            lease_id="lease-a",
            ttl=300,
            reservations_key=tracked,
            reservation_entry=entry,
        )
        == "reserved"
    )
    assert (
        await reserve_result_identity(
            redis=upstash_client,
            key=key,
            reservation=reservation,
            digest="abc",
            lease_id="lease-a",
            ttl=300,
            reservations_key=tracked,
            reservation_entry=entry,
        )
        == "retry"
    )
    assert (
        await reserve_result_identity(
            redis=upstash_client,
            key=key,
            reservation=reservation,
            digest="different",
            lease_id="lease-a",
            ttl=300,
            reservations_key=tracked,
            reservation_entry=entry,
        )
        == "busy"
    )
    await upstash_client.set(
        key, json.dumps({"status": "uploaded", "lease_id": "lease-a", "digest": "abc"})
    )
    assert (
        await reserve_result_identity(
            redis=upstash_client,
            key=key,
            reservation=reservation,
            digest="abc",
            lease_id="lease-a",
            ttl=300,
            reservations_key=tracked,
            reservation_entry=entry,
        )
        == "duplicate"
    )
    assert await upstash_client.sismember(tracked, entry)
