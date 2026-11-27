"""One-use GitHub device authorisation and opaque worker credentials."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import time
from contextlib import suppress
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from upstash_redis.asyncio import Redis

from .dependencies import (
    PROTOCOL_VERSION,
    fetch_with_retry,
    get_http,
    get_redis,
    read_json,
    require_env,
    require_protocol,
)
from .grants import UserGrants, WorkerIdentity, app_grant_enabled
from .state import redis_get_json

router = APIRouter()
_PREFIX = "euroeval:worker:"
_CREDENTIAL_TTL = 30 * 24 * 60 * 60
_COMPLETE = """
if redis.call('EXISTS',KEYS[1]) == 0 or redis.call('EXISTS',KEYS[2]) == 0
then return 0 end; return redis.call('DEL',KEYS[1])
"""
_RELEASE = """
if redis.call('GET',KEYS[1]) == ARGV[1]
then return redis.call('DEL',KEYS[1]) else return 0 end
"""


@router.post("/api/worker/auth/poll")
async def auth_poll(request: Request) -> JSONResponse:
    """Consume a device session exactly once after removing excess GitHub authority.

    Args:
        request:
            Device polling request.

    Returns:
        Pending instructions or a newly issued opaque worker credential.

    Raises:
        HTTPException:
            If polling, identity, scope, revocation or atomic consumption fails.
    """
    body = await read_json(request=request, limit=8192)
    require_protocol(body=body)
    session = body.get("session_id")
    if (
        not isinstance(session, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{12,}", session) is None
    ):
        raise HTTPException(status_code=400, detail="session_id is required.")
    redis = await get_redis(request=request)
    key = f"{_PREFIX}device:{session}"
    await enforce_rate_limit(
        redis=redis,
        key=f"{_PREFIX}limit:auth-poll:{session}",
        limit=120,
        window_seconds=900,
    )
    stored = await redis_get_json(redis=redis, key=key)
    if not isinstance(stored, dict):
        raise HTTPException(
            status_code=400,
            detail="Device authorisation has expired or was already used.",
        )
    device = stored
    next_poll = device.get("next_poll_at")
    if isinstance(next_poll, (int, float)) and next_poll > time.time() * 1000:
        seconds = math.ceil((next_poll - time.time() * 1000) / 1000)
        raise HTTPException(status_code=429, detail=f"Poll again in {seconds} seconds.")
    lock_key = f"{key}:completion-lock"
    token = secrets.token_urlsafe(nbytes=18)
    if not await redis.set(key=lock_key, value=token, ex=30, nx=True):
        raise HTTPException(
            status_code=429, detail="Authorization is being completed; retry shortly."
        )
    try:
        return await _complete_poll(
            request=request, redis=redis, key=key, device=device
        )
    finally:
        with suppress(Exception):
            await redis.execute(command=["EVAL", _RELEASE, "1", lock_key, token])


async def _complete_poll(
    request: Request, redis: Redis, key: str, device: dict[str, object]
) -> JSONResponse:
    """Finish the provider flow while the device completion lock is held.

    Args:
        request:
            Incoming request and shared HTTP pool.
        redis:
            Shared official Redis client.
        key:
            One-use device session key.
        device:
            Previously issued device configuration.

    Returns:
        Pending or authorised device result.

    Raises:
        HTTPException:
            If provider authority or one-use consumption cannot be established.
    """
    http = get_http(request=request)
    app_mode = bool(device.get("app_mode"))
    client_id = device.get("client_id") or require_env(name="GITHUB_OAUTH_CLIENT_ID")
    response = await fetch_with_retry(
        http=http,
        url="https://github.com/login/oauth/access_token",
        method="POST",
        headers={"accept": "application/json", "content-type": "application/json"},
        body={
            "client_id": client_id,
            "device_code": device.get("device_code"),
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            **({"repository_id": 386992479} if app_mode else {}),
        },
    )
    data = response.json()
    if data.get("error") in {"authorization_pending", "slow_down"}:
        interval = device.get("interval", 5)
        interval = int(interval) if isinstance(interval, (int, float)) else 5
        if data["error"] == "slow_down":
            interval = max(5, interval + 5)
        await redis.set(
            key=key,
            value=json.dumps(
                {
                    **device,
                    "interval": interval,
                    "next_poll_at": int(time.time() * 1000) + interval * 1000,
                }
            ),
            ex=900,
        )
        return JSONResponse(
            {
                "protocol_version": PROTOCOL_VERSION,
                "status": "pending",
                "retry_after": interval,
            },
            status_code=202,
            headers={"retry-after": str(interval)},
        )
    access = data.get("access_token")
    if not response.is_success or not access:
        await redis.delete(key)
        raise HTTPException(
            status_code=401, detail="GitHub device authorisation was denied or expired."
        )
    if app_mode and (
        not app_grant_enabled()
        or device.get("client_id") != os.getenv("VOLUNTEER_GITHUB_APP_CLIENT_ID")
        or device.get("repository_id") != 386992479
    ):
        raise HTTPException(
            status_code=503,
            detail="GitHub App grant configuration changed during authorization.",
        )
    user_response = await fetch_with_retry(
        http=http,
        url="https://api.github.com/user",
        headers={
            "accept": "application/vnd.github+json",
            "authorization": f"Bearer {access}",
            "x-github-api-version": "2022-11-28",
        },
    )
    if not user_response.is_success:
        await redis.delete(key)
        raise HTTPException(
            status_code=401, detail="GitHub could not identify this worker."
        )
    user = user_response.json()
    login = user.get("login")
    if not isinstance(login, str) or not login:
        await redis.delete(key)
        raise HTTPException(
            status_code=401, detail="GitHub could not identify this worker."
        )
    grants = UserGrants(http=http, redis=redis)
    if app_mode:
        repository_response = await fetch_with_retry(
            http=http,
            url="https://api.github.com/repos/EuroEval/EuroEval",
            headers={
                "accept": "application/vnd.github+json",
                "authorization": f"Bearer {access}",
                "x-github-api-version": "2022-11-28",
            },
        )
        repository = (
            repository_response.json() if repository_response.is_success else {}
        )
        if (
            repository.get("id") != 386992479
            or repository.get("full_name", "").lower() != "euroeval/euroeval"
        ):
            await redis.delete(key)
            with suppress(Exception):
                await grants.discard_user_access_token(access_token=access)
            raise HTTPException(
                status_code=403,
                detail="GitHub could not verify access to the configured repository.",
            )
        if not all(
            data.get(field)
            for field in ("refresh_token", "expires_in", "refresh_token_expires_in")
        ):
            raise HTTPException(
                status_code=401,
                detail="GitHub App authorization did not provide renewable tokens.",
            )
    else:
        secret = require_env(name="GITHUB_OAUTH_CLIENT_SECRET")
        basic = httpx.BasicAuth(username=str(client_id), password=secret)
        revoke_request = http.build_request(
            method="DELETE",
            url=f"https://api.github.com/applications/{client_id}/grant",
            json={"access_token": access},
            headers={"accept": "application/vnd.github+json"},
        )
        revoked = await http.send(request=revoke_request, auth=basic)
        if not revoked.is_success:
            raise HTTPException(
                status_code=503,
                detail=(
                    "GitHub grant revocation failed; no broker credential was issued."
                ),
            )
    credential = secrets.token_urlsafe(nbytes=32)
    digest = hashlib.sha256(credential.encode()).hexdigest()
    credential_key = f"{_PREFIX}credential:{digest}"
    record = {
        "contributor": login,
        "issued_at": datetime.now(tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
    }
    if app_mode:
        record["grant_mode"] = "app"
    if not await redis.set(
        key=credential_key, value=json.dumps(record), ex=_CREDENTIAL_TTL, nx=True
    ):
        raise HTTPException(
            status_code=503, detail="Unable to issue worker credential safely."
        )
    try:
        if app_mode:
            now = int(time.time() * 1000)
            await grants.retain_user_grant(
                access_token=access,
                refresh_token=data["refresh_token"],
                access_expires=now + data["expires_in"] * 1000,
                refresh_expires=now + data["refresh_token_expires_in"] * 1000,
                credential=credential,
                expected_login=login,
            )
        consumed = await redis.execute(
            command=["EVAL", _COMPLETE, "2", key, credential_key]
        )
        if consumed != 1:
            raise HTTPException(
                status_code=400,
                detail="Device authorisation has expired or was already used.",
            )
    except BaseException:
        if app_mode:
            with suppress(Exception):
                await grants.revoke_user_grant(
                    identity={"hash": digest, "contributor": login}
                )
            with suppress(Exception):
                await grants.discard_user_access_token(access_token=access)
        else:
            await redis.delete(credential_key)
        raise
    return JSONResponse(
        {
            "protocol_version": PROTOCOL_VERSION,
            "status": "authorised",
            "credential": credential,
            "github_login": login,
            "expires_in": _CREDENTIAL_TTL,
        }
    )


async def enforce_rate_limit(
    redis: Redis, key: str, limit: int, window_seconds: int
) -> None:
    """Enforce the existing fixed-window quota without changing its Redis keys.

    Args:
        redis:
            Shared official SDK client.
        key:
            Quota namespace.
        limit:
            Maximum requests in the window.
        window_seconds:
            Fixed window duration.

    Raises:
        HTTPException:
            If the request exceeds its quota.
    """
    bucket = f"{key}:{math.floor(time.time() / window_seconds)}"
    count = await redis.incr(key=bucket)
    if int(count) == 1:
        await redis.expire(key=bucket, seconds=window_seconds)
    if int(count) > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; retry later.")


@router.post("/api/worker/auth/revoke")
async def auth_revoke(request: Request) -> JSONResponse:
    """Revoke local authority, allowing encrypted remote-cleanup retries afterwards.

    Args:
        request:
            Credential or pending-revocation bearer request.

    Returns:
        Confirmed local revocation result.

    Raises:
        HTTPException:
            If no current credential or pending revocation belongs to the bearer.
    """
    credential = auth_credential(request=request)
    digest = hashlib.sha256(credential.encode()).hexdigest()
    redis = await get_redis(request=request)
    worker = await redis_get_json(redis=redis, key=f"{_PREFIX}credential:{digest}")
    identity: WorkerIdentity
    if isinstance(worker, dict) and worker.get("contributor"):
        identity = await authenticate(request=request, allow_legacy_grant=True)
    elif await redis.get(f"{_PREFIX}user-grant:{digest}:revocation"):
        identity = {"hash": digest, "contributor": ""}
    else:
        raise HTTPException(
            status_code=401, detail="Worker credential is invalid or revoked."
        )
    await UserGrants(http=get_http(request=request), redis=redis).revoke_user_grant(
        identity=identity
    )
    return JSONResponse({"protocol_version": PROTOCOL_VERSION, "status": "revoked"})


def auth_credential(request: Request) -> str:
    """Read the deployed case-sensitive bearer credential form.

    Args:
        request:
            Incoming request.

    Returns:
        Unhashed worker credential.

    Raises:
        HTTPException:
            If authorisation is missing or malformed.
    """
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer ") or len(header) < 10:
        raise HTTPException(
            status_code=401,
            detail="Use the worker credential as Authorization: Bearer <credential>.",
        )
    return header[7:].strip()


async def authenticate(
    request: Request, allow_legacy_grant: bool = False
) -> WorkerIdentity:
    """Authenticate cached authority without allowing old grants in App mode.

    Args:
        request:
            Incoming authenticated request.
        allow_legacy_grant (optional):
            Permit local revocation of old credentials, defaulting to false.

    Returns:
        Verified contributor and credential hash.

    Raises:
        HTTPException:
            If authority is invalid, revoked, or requires App renewal.
    """
    digest = hashlib.sha256(auth_credential(request=request).encode()).hexdigest()
    worker = await redis_get_json(
        redis=await get_redis(request=request), key=f"{_PREFIX}credential:{digest}"
    )
    if (
        not isinstance(worker, dict)
        or not isinstance(worker.get("contributor"), str)
        or not worker["contributor"]
    ):
        raise HTTPException(
            status_code=401, detail="Worker credential is invalid or revoked."
        )
    if (
        not allow_legacy_grant
        and app_grant_enabled()
        and worker.get("grant_mode") != "app"
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Worker authorization must be renewed for GitHub App access.",
                "code": "github_grant_invalid",
            },
        )
    return {"hash": digest, "contributor": worker["contributor"]}


@router.post("/api/worker/auth/start")
async def auth_start(request: Request) -> JSONResponse:
    """Begin bounded device authorisation without accepting spoofed proxy identities.

    Args:
        request:
            Device-authorisation request.

    Returns:
        Device session, verification instructions and polling interval.

    Raises:
        HTTPException:
            If GitHub rejects or incompletely describes the device flow.
    """
    redis = await get_redis(request=request)
    peer = request.client.host if request.client else "unknown"
    peer_hash = hashlib.sha256(peer.encode()).hexdigest()
    await enforce_rate_limit(
        redis=redis,
        key=f"{_PREFIX}limit:auth-start:{peer_hash}",
        limit=5,
        window_seconds=3600,
    )
    body = await read_json(request=request, limit=4096)
    require_protocol(body=body)
    app_mode = app_grant_enabled()
    http = get_http(request=request)
    grants = UserGrants(http=http, redis=redis)
    if app_mode:
        await grants.verify_app_installation()
    client_id = require_env(
        name="VOLUNTEER_GITHUB_APP_CLIENT_ID" if app_mode else "GITHUB_OAUTH_CLIENT_ID"
    )
    response = await fetch_with_retry(
        http=http,
        url="https://github.com/login/device/code",
        method="POST",
        headers={"accept": "application/json", "content-type": "application/json"},
        body={
            "client_id": client_id,
            "scope": "offline_access" if app_mode else "read:user",
        },
    )
    if not response.is_success:
        raise HTTPException(
            status_code=503,
            detail=(
                f"GitHub device authorisation failed with HTTP {response.status_code}."
            ),
        )
    data = response.json()
    if not all(
        data.get(key) for key in ("device_code", "user_code", "verification_uri")
    ):
        raise HTTPException(
            status_code=503,
            detail="GitHub returned an incomplete device authorisation response.",
        )
    session = secrets.token_urlsafe(nbytes=18)
    ttl = max(60, min(900, int(data.get("expires_in") or 600)))
    device = {
        "device_code": data["device_code"],
        "client_id": client_id,
        "interval": data.get("interval") or 5,
        "app_mode": app_mode,
    }
    if app_mode:
        device["repository_id"] = 386992479
    await redis.set(key=f"{_PREFIX}device:{session}", value=json.dumps(device), ex=ttl)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session,
        "user_code": data["user_code"],
        "verification_uri": data["verification_uri"],
        "interval": data.get("interval") or 5,
        "expires_in": ttl,
    }
    if "verification_uri_complete" in data:
        result["verification_uri_complete"] = data["verification_uri_complete"]
    return JSONResponse(result)


def coordinator_secret(request: Request) -> None:
    """Validate coordinator authority independently from worker credentials.

    Args:
        request:
            Incoming coordinator request.
    """
    _secret(
        request=request,
        name="WORKER_COORDINATOR_SECRET",
        header="x-coordinator-secret",
        label="Coordinator",
    )


def _secret(request: Request, name: str, header: str, label: str) -> None:
    """Use constant-time byte comparison for configured role secrets.

    Args:
        request:
            Incoming role-authenticated request.
        name:
            Required secret configuration name.
        header:
            Role-specific header, falling back to bearer authorisation.
        label:
            Role used in the public failure message.

    Raises:
        HTTPException:
            If role authentication fails.
    """
    expected = require_env(name=name)
    supplied = request.headers.get(header) or auth_credential(request=request)
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail=f"{label} authentication failed.")


def promotion_secret(request: Request) -> None:
    """Validate the separate canonical-publication authority.

    Args:
        request:
            Incoming promotion request.
    """
    _secret(
        request=request,
        name="VOLUNTEER_PROMOTION_SECRET",
        header="x-promotion-secret",
        label="Promotion",
    )
