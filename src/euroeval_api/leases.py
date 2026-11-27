"""Shared lease schema and live-lease ownership authority."""

from __future__ import annotations

import math
import time
import typing as t

from fastapi import HTTPException
from upstash_redis.asyncio import Redis

from . import markers, state
from .model import ModelMetadataEvidence
from .policy import ModelType


class LeaseScope(t.TypedDict):
    """Task and identity coverage persisted with a lease."""

    policy_version: str
    language_group: str
    task_groups: list[str]
    warnings: list[str]
    allowed_identity_suffix_sets: t.NotRequired[list[list[str]]]
    identity_suffixes: t.NotRequired[list[str]]
    original_identity_suffix_sets: t.NotRequired[list[list[str]]]
    covered_identity_suffixes: t.NotRequired[list[str]]
    count: t.NotRequired[int]


class Lease(state.LeaseRecord):
    """Complete lease exchanged with worker lifecycle routes."""

    worker: str
    contributor: str
    model_id: str
    model_revision: str
    euroeval_version: str
    image_digest: str
    worker_version: str
    gpu_memory_utilisation: float
    selected_gpu_index: int
    selected_gpu_uuid: str
    model_type: ModelType
    model_metadata: ModelMetadataEvidence
    expected_scope: LeaseScope
    ownership_mode: t.NotRequired[t.Literal["assignee", "public"]]
    record_revision_required: t.NotRequired[bool]
    released: t.NotRequired[bool]
    result_count: t.NotRequired[int]
    contamination_canary: t.NotRequired[dict[str, object]]


async def require_live_lease(redis: Redis, lease_id: str, contributor: str) -> Lease:
    """Load a live lease owned by the given contributor.

    Args:
        redis: Official asynchronous Redis client.
        lease_id: Opaque lease identifier.
        contributor: Authenticated contributor identity.

    Returns:
        The complete persisted lease schema.

    Raises:
        HTTPException: The lease is absent, expired, released, or not owned.
    """
    record = await state.get_lease_by_id(redis=redis, lease_id=lease_id)
    if record is None:
        raise HTTPException(status_code=409, detail="Lease not found or not owned.")
    lease = t.cast(Lease, record)
    if (
        lease.get("released") is True
        or not isinstance(lease.get("contributor"), str)
        or lease["contributor"].casefold() != contributor.casefold()
    ):
        raise HTTPException(status_code=409, detail="Lease not found or not owned.")

    expires_at = markers.timestamp(lease["expires_at"])
    if not math.isfinite(expires_at) or expires_at <= time.time() * 1000:
        raise HTTPException(status_code=409, detail="Lease not found or not owned.")
    return lease
