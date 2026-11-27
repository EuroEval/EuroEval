"""Public outcomes of real Redis Lua lease and reservation transactions."""

from __future__ import annotations

import hashlib
import json
import typing as t

import pytest
import redis as redis_module
from fastapi import HTTPException
from upstash_redis.asyncio import Redis

from euroeval_api.state import (
    LeaseRecord,
    abort_result_identity,
    acquire_issue_mutex,
    acquire_renewable_issue_mutex,
    complete_result_identity,
    delete_lease,
    get_lease_by_id,
    get_lease_for_issue,
    identity_key,
    mutex_key,
    put_lease,
    reclaim_expired_lease,
    release_result_reservations,
    reserve_result_identity,
    result_reservations_key,
)


@pytest.mark.anyio
async def test_expired_reclaim_preserves_replacements(upstash_client: Redis) -> None:
    """Reclaim matching identities, not a replaced identity or subsequent live lease.

    Args:
        upstash_client: SDK connected to real local Redis.
    """
    owner = lease(lease_id="expired", expires_at="2000-01-01T00:00:00Z")
    digest = hashlib.sha256(b"result bytes").hexdigest()
    assert await put_lease(redis=upstash_client, lease=owner)
    first, second = "model/first-dataset", "model/second-dataset"
    assert (
        await reserve(
            client=upstash_client, identity=first, owner="expired", digest=digest
        )
        == "reserved"
    )
    assert (
        await reserve(
            client=upstash_client, identity=second, owner="expired", digest=digest
        )
        == "reserved"
    )
    await abort_result_identity(
        redis=upstash_client,
        key=identity_key(identity=second),
        digest=digest,
        lease_id="expired",
    )
    assert (
        await reserve(
            client=upstash_client, identity=second, owner="replacement", digest=digest
        )
        == "reserved"
    )
    assert await reclaim_expired_lease(redis=upstash_client, lease=owner)
    assert await upstash_client.get(identity_key(identity=first)) is None
    assert await upstash_client.get(identity_key(identity=second)) is not None
    assert await get_lease_by_id(redis=upstash_client, lease_id="expired") is None
    replacement = lease(lease_id="live")
    assert await put_lease(redis=upstash_client, lease=replacement)
    assert not await reclaim_expired_lease(redis=upstash_client, lease=owner)
    assert (
        await get_lease_for_issue(redis=upstash_client, issue=42, language="da")
        == replacement
    )


def lease(lease_id: str, expires_at: str = "2099-01-01T00:00:00Z") -> LeaseRecord:
    """Create opaque persisted lease evidence.

    Args:
        lease_id: Lease owner.
        expires_at (optional): Lease expiration used by ownership recovery.

    Returns:
        Persistable lease with an extension the broker must retain.
    """
    return t.cast(
        LeaseRecord,
        {
            "lease_id": lease_id,
            "issue_number": 42,
            "language": "da",
            "expires_at": expires_at,
            "opaque": {"kept": True},
        },
    )


async def reserve(client: Redis, identity: str, owner: str, digest: str) -> str:
    """Reserve a result through the production public transaction.

    Args:
        client: Official SDK client.
        identity: Logical result identity.
        owner: Owning lease.
        digest: Exact result bytes digest.

    Returns:
        Reservation outcome.
    """
    return await reserve_result_identity(
        redis=client,
        key=identity_key(identity=identity),
        reservation=json.dumps(
            {
                "status": "uploading",
                "lease_id": owner,
                "digest": digest,
                "identity": identity,
            }
        ),
        digest=digest,
        lease_id=owner,
        ttl=300,
        reservations_key=result_reservations_key(lease_id=owner),
        reservation_entry=json.dumps({"identity": identity, "digest": digest}),
    )


@pytest.mark.anyio
async def test_lease_fencing_and_released_tombstone(upstash_client: Redis) -> None:
    """Retain opaque fields and never release another lease through an old retry.

    Args:
        upstash_client: SDK connected to real local Redis.
    """
    owner = lease(lease_id="owner")
    foreign = lease(lease_id="foreign")
    assert await put_lease(redis=upstash_client, lease=owner)
    assert await put_lease(redis=upstash_client, lease=owner)
    assert not await put_lease(redis=upstash_client, lease=foreign)
    assert await get_lease_by_id(redis=upstash_client, lease_id="owner") == owner
    await delete_lease(redis=upstash_client, lease=owner)
    tombstone = await get_lease_by_id(redis=upstash_client, lease_id="owner")
    assert tombstone is not None and tombstone.get("released") is True
    assert await put_lease(redis=upstash_client, lease=foreign)
    await delete_lease(redis=upstash_client, lease=owner)
    assert (
        await get_lease_for_issue(redis=upstash_client, issue=42, language="da")
        == foreign
    )


@pytest.mark.anyio
async def test_mutex_refuses_lost_ownership(
    upstash_client: Redis, redis_client: redis_module.Redis
) -> None:
    """A stale mutex cannot authorize a mutation or delete its replacement.

    Args:
        upstash_client: SDK connected to real local Redis.
        redis_client: Server-side client used to expire the original token.
    """
    mutex = await acquire_renewable_issue_mutex(redis=upstash_client, issue=42)
    assert mutex is not None
    try:
        await mutex.assert_owned(timeout=1)
        redis_client.expire(mutex_key(issue=42), 0)
        replacement = await acquire_issue_mutex(redis=upstash_client, issue=42)
        assert replacement is not None
        with pytest.raises(HTTPException) as failure:
            await mutex.assert_owned(timeout=1)
        assert failure.value.status_code == 409
        await mutex.release(timeout=1)
        assert await upstash_client.get(mutex_key(issue=42)) == replacement
    finally:
        await mutex.release(timeout=1)


@pytest.mark.anyio
async def test_reservation_outcomes_and_atomic_cleanup(upstash_client: Redis) -> None:
    """Distinguish retry/duplicate/conflict and abort cleanup on any foreign owner.

    Args:
        upstash_client: SDK connected to real local Redis.
    """
    digest = hashlib.sha256(b"result bytes").hexdigest()
    first, second = "model/first-dataset", "model/second-dataset"
    assert (
        await reserve(
            client=upstash_client, identity=first, owner="owner", digest=digest
        )
        == "reserved"
    )
    assert (
        await reserve(
            client=upstash_client, identity=first, owner="owner", digest=digest
        )
        == "retry"
    )
    assert (
        await reserve(
            client=upstash_client, identity=first, owner="owner", digest="different"
        )
        == "busy"
    )
    assert await complete_result_identity(
        redis=upstash_client,
        key=identity_key(identity=first),
        reservation=json.dumps(
            {"status": "uploaded", "lease_id": "owner", "digest": digest}
        ),
        digest=digest,
        lease_id="owner",
        ttl=300,
    )
    assert (
        await reserve(
            client=upstash_client, identity=first, owner="owner", digest=digest
        )
        == "duplicate"
    )
    assert (
        await reserve(
            client=upstash_client, identity=second, owner="replacement", digest=digest
        )
        == "reserved"
    )
    records = [
        {"identity": first, "digest": digest},
        {"identity": second, "digest": digest},
    ]
    assert not await release_result_reservations(
        redis=upstash_client, records=records, lease_id="owner"
    )
    assert await upstash_client.get(identity_key(identity=first)) is not None
    assert await upstash_client.get(identity_key(identity=second)) is not None
    assert await release_result_reservations(
        redis=upstash_client, records=records[:1], lease_id="owner"
    )
    assert await upstash_client.get(identity_key(identity=first)) is None
    assert await upstash_client.get(identity_key(identity=second)) is not None
