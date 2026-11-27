"""Worker result submission endpoint."""

from __future__ import annotations

import hashlib
import json
import math
import time
import typing as t

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic_core import from_json
from upstash_redis.asyncio import Redis

from .auth import authenticate, enforce_rate_limit
from .dependencies import (
    PROTOCOL_VERSION,
    get_http,
    get_redis,
    read_json,
    require_protocol,
)
from .eee import validate_record
from .github import GitHub, require_lease_ownership
from .leases import Lease, require_live_lease
from .markers import timestamp
from .staging import upload_staging
from .state import (
    abort_result_identity,
    acquire_renewable_issue_mutex,
    complete_result_identity,
    redis_get_json,
    release_result_reservation,
    reserve_result_identity,
)
from .state import identity_key as record_identity_key

router = APIRouter()


class StoredIdentity(t.TypedDict, total=False):
    """Persisted reservation state."""

    digest: str
    lease_id: str
    status: str
    issue_number: int
    language: str
    path: str
    warnings: list[str]


@router.post("/api/worker/result")
async def result(request: Request) -> JSONResponse:
    """Validate and stage one exact worker result record.

    Args:
        request:
            Incoming result request.

    Returns:
        Protocol receipt for the uploaded or duplicate record.

    Raises:
        HTTPException:
            If authentication, validation, ownership, or upload fails.
    """
    identity = await authenticate(request=request)
    redis = await get_redis(request=request)
    await enforce_rate_limit(
        redis=redis,
        key=f"euroeval:worker:limit:result:{identity['hash']}",
        limit=500,
        window_seconds=3600,
    )
    body = await read_json(request=request, limit=2 * 1024 * 1024)
    require_protocol(body=body)
    fields = (
        "lease_id",
        "issue_number",
        "language",
        "model_id",
        "model_revision",
        "euroeval_version",
        "image_digest",
        "worker_version",
        "record_json",
        "digest",
    )
    for field in fields:
        if field not in body:
            raise HTTPException(status_code=400, detail=f"{field} is required.")
    issue_number = body.get("issue_number")
    if (
        any(
            not isinstance(body.get(field), str)
            for field in fields
            if field != "issue_number"
        )
        or isinstance(issue_number, bool)
        or not isinstance(issue_number, int | float)
        or abs(issue_number) > 2**53 - 1
        or not math.isfinite(issue_number)
        or int(issue_number) != issue_number
    ):
        raise HTTPException(
            status_code=400, detail="Result contract fields have invalid types."
        )
    lease_id = t.cast(str, body["lease_id"])
    lease = await require_live_lease(
        redis=redis, lease_id=lease_id, contributor=identity["contributor"]
    )
    github = GitHub(http=get_http(request=request))
    issue = await github.get_issue(number=lease["issue_number"])
    require_lease_ownership(issue=issue, lease=lease)
    _verify_contract(body=body, lease=lease)
    record_json = t.cast(str, body["record_json"])
    digest = hashlib.sha256(record_json.encode("utf-8")).hexdigest()
    if body["digest"] != digest:
        raise HTTPException(
            status_code=422, detail="digest does not match the UTF-8 record_json bytes."
        )
    try:
        record: object = from_json(record_json, allow_inf_nan=False)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422, detail="record_json must be valid JSON."
        ) from None
    checked = validate_record(
        record=record,
        model_id=lease["model_id"],
        revision=lease["model_revision"],
        language=lease["language"],
        euroeval_version=lease["euroeval_version"],
        model_type=lease["model_type"],
        require_revision=lease.get("record_revision_required") is True,
    )
    identity_key = record_identity_key(identity=checked["identity"])
    path = f"volunteer/submissions/{lease_id}/results/{digest}.json"
    existing = await redis_get_json(redis=redis, key=identity_key)
    if existing and existing.get("digest") != digest:
        raise HTTPException(
            status_code=409,
            detail="This canonical record identity already has a different digest.",
        )
    if existing and existing.get("status") == "uploaded":
        warnings = existing.get("warnings")
        if not isinstance(warnings, list):
            warnings = checked["warnings"]
        await _track(
            redis=redis,
            lease=lease,
            digest=digest,
            identity=checked["identity"],
            path=t.cast(str, existing.get("path") or path),
            failed=checked["failed"],
            warnings=warnings,
        )
        return _receipt(
            status="duplicate",
            digest=digest,
            identity=checked["identity"],
            path=t.cast(str, existing.get("path") or path),
            warnings=warnings,
        )
    reservation: StoredIdentity = {
        "digest": digest,
        "lease_id": lease_id,
        "status": "uploading",
        "issue_number": lease["issue_number"],
        "language": lease["language"],
        "path": path,
        "warnings": checked["warnings"],
    }
    mutex = await acquire_renewable_issue_mutex(
        redis=redis, issue=lease["issue_number"]
    )
    if mutex is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Issue is busy; retry result submission.",
                "code": "issue_busy",
            },
        )
    try:
        await mutex.assert_owned()
        require_lease_ownership(
            issue=await github.get_issue(number=lease["issue_number"]), lease=lease
        )
        claim = await reserve_result_identity(
            redis=redis,
            key=identity_key,
            reservation=json.dumps(
                reservation, ensure_ascii=False, separators=(",", ":")
            ),
            digest=digest,
            lease_id=lease_id,
            ttl=86400,
            reservations_key=f"euroeval:worker:reservations:{lease_id}",
            reservation_entry=json.dumps(
                {"digest": digest, "identity": checked["identity"]},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    finally:
        await mutex.release()
    if claim in {"busy", "full", "duplicate"}:
        raise HTTPException(status_code=409, detail=f"Result reservation {claim}.")
    completed = False
    try:
        await upload_staging(
            http=get_http(request=request), path=path, content=record_json
        )
        _check_expiry(lease=lease)
        completion = await acquire_renewable_issue_mutex(
            redis=redis, issue=lease["issue_number"]
        )
        if completion is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Issue is busy; retry result submission.",
                    "code": "issue_busy",
                },
            )
        try:
            await completion.assert_owned()
            require_lease_ownership(
                issue=await github.get_issue(number=lease["issue_number"]), lease=lease
            )
            reservation["status"] = "uploaded"
            if not await complete_result_identity(
                redis=redis,
                key=identity_key,
                reservation=json.dumps(
                    reservation, ensure_ascii=False, separators=(",", ":")
                ),
                digest=digest,
                lease_id=lease_id,
                ttl=30 * 86400,
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Result reservation was replaced during upload; retry safely."
                    ),
                )
            completed = True
            await completion.assert_owned()
            require_lease_ownership(
                issue=await github.get_issue(number=lease["issue_number"]), lease=lease
            )
            await _track(
                redis=redis,
                lease=lease,
                digest=digest,
                identity=checked["identity"],
                path=path,
                failed=checked["failed"],
                warnings=checked["warnings"],
                expiry=True,
            )
        finally:
            await completion.release()
    except BaseException as error:
        assignment_lost = (
            isinstance(error, HTTPException)
            and isinstance(error.detail, dict)
            and error.detail.get("code") == "lease_assignment_lost"
        )
        if completed and assignment_lost:
            await release_result_reservation(
                redis=redis, key=identity_key, digest=digest, lease_id=lease_id
            )
        else:
            await abort_result_identity(
                redis=redis, key=identity_key, digest=digest, lease_id=lease_id
            )
        raise
    return _receipt(
        status="uploaded",
        digest=digest,
        identity=checked["identity"],
        path=path,
        warnings=checked["warnings"],
        status_code=201,
    )


def _check_expiry(lease: Lease) -> None:
    """Reject uploads whose original lease expired during staging.

    Args:
        lease:
            Original authorised lease.

    Raises:
        HTTPException:
            If staging outlived its authorisation.
    """
    if timestamp(value=lease["expires_at"]) <= time.time() * 1000:
        raise HTTPException(
            status_code=409,
            detail="Lease is absent, expired, or belongs to another contributor.",
        )


def _receipt(
    status: str,
    digest: str,
    identity: str,
    path: str,
    warnings: list[str],
    status_code: int = 200,
) -> JSONResponse:
    """Build a protocol-compatible receipt.

    Args:
        status:
            Upload or duplicate outcome.
        digest:
            Exact content digest.
        identity:
            Canonical record identity.
        path:
            Private staging location.
        warnings:
            Validation findings.
        status_code (optional):
            HTTP outcome, defaulting to 200.

    Returns:
        Worker-facing receipt.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "protocol_version": PROTOCOL_VERSION,
            "status": status,
            "digest": digest,
            "identity": identity,
            "path": path,
            "warnings": warnings,
        },
    )


async def _track(
    redis: Redis,
    lease: Lease,
    digest: str,
    identity: str,
    path: str,
    failed: int,
    warnings: list[str],
    expiry: bool = False,
) -> None:
    """Record result and reservation set members in Node-compatible JSON.

    Args:
        redis:
            Authoritative result index.
        lease:
            Current lease.
        digest:
            Content hash.
        identity:
            Canonical result identity.
        path:
            Private staging location.
        failed:
            Failed instance count.
        warnings:
            Validation findings.
        expiry (optional):
            Whether to extend result index lifetimes, defaulting to false.
    """
    entry = json.dumps(
        {
            "digest": digest,
            "identity": identity,
            "path": path,
            "failed": failed,
            "warnings": warnings,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    reservation = json.dumps(
        {"digest": digest, "identity": identity},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    await redis.execute(
        command=["SADD", f"euroeval:worker:results:{lease['lease_id']}", entry]
    )
    await redis.execute(
        command=[
            "SADD",
            f"euroeval:worker:reservations:{lease['lease_id']}",
            reservation,
        ]
    )
    if expiry:
        await redis.execute(
            command=[
                "EXPIRE",
                f"euroeval:worker:results:{lease['lease_id']}",
                "2592000",
            ]
        )
        await redis.execute(
            command=[
                "EXPIRE",
                f"euroeval:worker:reservations:{lease['lease_id']}",
                "2592000",
            ]
        )


def _verify_contract(body: dict[str, object], lease: Lease) -> None:
    """Compare every submitted lease field exactly.

    Args:
        body:
            Submitted result contract.
        lease:
            Authoritative lease metadata.

    Raises:
        HTTPException:
            If the contract differs from the lease.
    """
    expected = {
        "issue_number": lease["issue_number"],
        "language": lease["language"],
        "model_id": lease["model_id"],
        "model_revision": lease["model_revision"],
        "euroeval_version": lease["euroeval_version"],
        "image_digest": lease["image_digest"],
        "worker_version": lease["worker_version"],
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise HTTPException(
            status_code=422,
            detail="Result contract does not exactly match the active lease.",
        )
