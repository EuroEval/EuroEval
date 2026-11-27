"""Coordinator endpoints for acquiring and maintaining issue mutexes."""

from __future__ import annotations

import typing as t

from fastapi import APIRouter, HTTPException, Request

from .auth import coordinator_secret
from .dependencies import PROTOCOL_VERSION, get_redis, read_json, require_protocol
from .state import (
    ISSUE_MUTEX_TTL,
    acquire_issue_mutex,
    release_issue_mutex,
    renew_issue_mutex,
)

router = APIRouter()


@router.post("/api/worker/coordinator-lock")
async def coordinator_lock(request: Request) -> dict[str, object]:
    """Acquire the shared mutex for an evaluation issue.

    Args:
        request: Incoming coordinator request.

    Returns:
        Lock token and protocol metadata.

    Raises:
        HTTPException: If authentication, validation, or lock acquisition fails.
    """
    body = await _body(request=request)
    issue, _ = _issue_number(body=body, token_required=False)
    token = await acquire_issue_mutex(
        redis=await get_redis(request=request), issue=issue
    )
    if token is None:
        raise HTTPException(
            status_code=409, detail="Issue is busy; retry coordinator lock."
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "issue_number": issue,
        "token": token,
        "expires_in": ISSUE_MUTEX_TTL,
    }


async def _body(request: Request) -> dict[str, object]:
    """Authenticate a coordinator request and parse its protocol body.

    Args:
        request: Incoming HTTP request.

    Returns:
        Validated protocol request body.
    """
    coordinator_secret(request=request)
    body = await read_json(request=request, limit=8 * 1024)
    require_protocol(body=body)
    return body


def _issue_number(
    body: dict[str, object], *, token_required: bool
) -> tuple[int, str | None]:
    """Validate lock identity and, where required, its owner token.

    Args:
        body: Parsed request object.
        token_required: Whether a non-empty token must be supplied.

    Returns:
        Validated issue number and optional token.

    Raises:
        HTTPException: If required lock fields are invalid.
    """
    issue = body.get("issue_number")
    token = body.get("token")
    if type(issue) is not int or issue < 1:
        raise HTTPException(
            status_code=400, detail="issue_number must be a positive integer."
        )
    if token_required and (not isinstance(token, str) or not token):
        raise HTTPException(
            status_code=400, detail="issue_number and token are required."
        )
    return issue, token if isinstance(token, str) else None


@router.post("/api/worker/coordinator-release")
async def coordinator_release(request: Request) -> dict[str, object]:
    """Release a mutex only when its opaque token owns the key.

    Args:
        request: Incoming coordinator request.

    Returns:
        Release status and protocol metadata.
    """
    body = await _body(request=request)
    issue, token = _issue_number(body=body, token_required=True)
    await release_issue_mutex(
        redis=await get_redis(request=request), issue=issue, token=t.cast(str, token)
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "released",
        "issue_number": issue,
    }


@router.post("/api/worker/coordinator-renew")
async def coordinator_renew(request: Request) -> dict[str, object]:
    """Renew a mutex only while its token still owns it.

    Args:
        request: Incoming coordinator request.

    Returns:
        Renewal status and protocol metadata.

    Raises:
        HTTPException: If authentication, validation, or ownership fails.
    """
    body = await _body(request=request)
    issue, token = _issue_number(body=body, token_required=True)
    if not await renew_issue_mutex(
        redis=await get_redis(request=request), issue=issue, token=t.cast(str, token)
    ):
        raise HTTPException(
            status_code=409, detail="Coordinator lock was lost or replaced."
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "renewed",
        "issue_number": issue,
        "expires_in": ISSUE_MUTEX_TTL,
    }
